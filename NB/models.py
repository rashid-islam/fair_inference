import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions import Normal, Gamma, Uniform

device = torch.device("cpu")

#%% A fully connected deep neural network
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,output_size,hidden,d_prob,acts):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden[0])
        self.fc2 = nn.Linear(hidden[0],hidden[1])
        self.out_fc = nn.Linear(hidden[1],output_size)
        self.out_bn = nn.BatchNorm1d(output_size) 
         
        self.d_out = nn.Dropout(d_prob)

        if acts == 'rectify':
            self.act  = nn.ReLU()
        elif acts == 'selu':
            self.act = nn.SELU()
        elif acts == 'elu':
            self.act = nn.ELU()
        elif acts == 'leaky':
            self.act = nn.LeakyReLU()
        elif acts == 'softplus':
            self.act = nn.Softplus()
        elif acts == 'tanh':
            self.act = nn.Tanh()
        elif acts == 'prelu':
            self.act = nn.PReLU()
        
    def forward(self,x):
        out = self.d_out(self.act(self.fc1(x)))
        out = self.d_out(self.act(self.fc2(out)))
        out = self.out_bn(self.out_fc(out))
        return out 
    
#%% Gumbel-Sigmoid for discrete latent variables
# Sample from Gumbel Softmax distribution - from Yongfei Yan's Github page
# source: https://github.com/YongfeiYan/Gumbel_Softmax_VAE
# source: https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_sigmoid.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if device.type == "cuda":
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature,latent_size,categorical_size, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1,latent_size * categorical_size)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_size * categorical_size)

#%% Discrete latent variable models 

#%% Unsupervised Naive Bayes Model using variational inference
class NB(nn.Module):
    def __init__(self,input_size, latent_size, categorical_size, encode_hidden ,d_prob, acts):
        super(NB,self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.categorical_size = categorical_size
        
        # define prior on model parameters
        # informed prior on the Logit-normal distribution
        self.mu0_prior = Normal(-2.0,1.0) # latent class = 0        
        self.mu1_prior = Normal(2.0,1.0) # latent class = 1
        
        self.sigma_prior = Gamma(1.0,2.0)
        
        # initialize model parameters
        self.mu_z0 = nn.Parameter(torch.randn(1,input_size, requires_grad=True))
        self.mu_z1 = nn.Parameter(torch.randn(1,input_size, requires_grad=True)) # std = torch.exp(0.5*logvar) or Variance = std^2

            # modify to nn.Parameter(Uniform(-1, 1).sample([input_a]).view(1,-1), requires_grad=True)
        self.logvar_z0 = nn.Parameter(Uniform(-1, 1).sample([1,input_size]), requires_grad=True) 
        self.logvar_z1 = nn.Parameter(Uniform(-1, 1).sample([1,input_size]), requires_grad=True) 
        
        self.encoder_net = NeuralNetwork(input_size,latent_size*categorical_size,encode_hidden,d_prob,acts)
        
    def reparameterize_latent(self, mu, sigma): 
        eps = torch.randn_like(sigma)
        sample = torch.softmax(mu + (sigma*eps), dim=1) # drawing sample from logistic normal distribution
        return sample
    
    def reconstruct_x(self,x):  
        eps_z0 = torch.randn_like(x)
        eps_z1 = torch.randn_like(x)
        # convert log variance to standard deviation
        sigma_z0 = torch.exp(0.5*self.logvar_z0)
        sigma_z1 = torch.exp(0.5*self.logvar_z1)
        # reparameterization trick
        beta_z0 = self.mu_z0 + sigma_z0*eps_z0
        beta_z1 = self.mu_z1 + sigma_z1*eps_z1
        return beta_z0, beta_z1, self.mu_z0, self.mu_z1, sigma_z0, sigma_z1
    
    def parameter_prior_log_prob(self,mu_z0,mu_z1,sigma_z0,sigma_z1):
        # log prior for model parameters
        log_prior_param = self.mu0_prior.log_prob(mu_z0).sum() + self.mu1_prior.log_prob(mu_z1).sum() + self.sigma_prior.log_prob(sigma_z0).sum() + self.sigma_prior.log_prob(sigma_z1).sum() 
        return log_prior_param
    
    def forward(self,x,temp):
        q_z_pi = self.encoder_net(x)
        q_z_y = q_z_pi.view(q_z_pi.size(0), self.latent_size, self.categorical_size)
        q_z = gumbel_softmax(q_z_y, temp, self.latent_size, self.categorical_size)
        
        beta_z0, beta_z1, mu_z0, mu_z1, sigma_z0, sigma_z1 = self.reconstruct_x(x)
        
        log_prior_param = self.parameter_prior_log_prob(mu_z0,mu_z1,sigma_z0,sigma_z1)
        return q_z, beta_z0, beta_z1, mu_z0, mu_z1, sigma_z0, sigma_z1, log_prior_param  

#%% Categorical VAE using Gumbel-Softmax along with Bernoulli MLP as decoder
class  BernoulliVAE(nn.Module):
    def __init__(self,input_size,latent_size, categorical_size,encode_hidden,d_prob, acts):
        super(BernoulliVAE,self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.categorical_size = categorical_size
        
        
        # initialize model parameters
        self.encoder_net = NeuralNetwork(input_size,latent_size*categorical_size,encode_hidden,d_prob,acts)
        self.decoder_net = NeuralNetwork(latent_size*categorical_size,input_size,list(reversed(encode_hidden)),d_prob,acts)
        
    def encoder(self,x):
        out = self.encoder_net(x)
        return out
    
    def decoder(self,x):  
        out = self.decoder_net(x)
        return out
    
    def forward(self,x, temp):
        q = self.encoder(x)
        q_y = q.view(q.size(0), self.latent_size, self.categorical_size)
        z = gumbel_softmax(q_y, temp,self.latent_size, self.categorical_size)
        out = self.decoder(z)
        return z, out

#%% unsupervised variational fair autoencoder (VFAE)
class  BernoulliVFAE(nn.Module):
    def __init__(self,input_size,latent_size,categorical_size,group_size,encode_hidden,d_prob, acts):
        super(BernoulliVFAE,self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.categorical_size = categorical_size
        
        
        # initialize model parameters
        self.encoder_net = NeuralNetwork(input_size,latent_size*categorical_size,encode_hidden,d_prob,acts)
        self.decoder_net = NeuralNetwork((latent_size*categorical_size)+group_size,input_size,list(reversed(encode_hidden)),d_prob,acts)
        
    def encoder(self,x):
        out = self.encoder_net(x)
        return out
    
    def decoder(self,x):  
        out = self.decoder_net(x)
        return out
    
    def forward(self,x, s, temp):
        q = self.encoder(x)
        q_y = q.view(q.size(0), self.latent_size, self.categorical_size)
        z = gumbel_softmax(q_y, temp,self.latent_size, self.categorical_size)
        out = self.decoder(torch.cat((z, s), dim=1))
        return z, out
