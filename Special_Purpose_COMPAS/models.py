import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions import Normal, Gamma, Uniform

from torch.nn.init import xavier_uniform_, xavier_normal_

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


#%% Special purpose (SP) latent variable model using variational inference for criminal justice (on top of COMPAS system)
class SP(nn.Module):
    def __init__(self,input_z, input_a, input_c, input_t, latent_size, cat_size_z, cat_size_u, encode_hidden, d_prob, acts):
        super(SP,self).__init__()
        
        self.input_z = input_z
        self.input_a = input_a
        self.input_c = input_c
        self.input_t = input_t
        self.latent_size = latent_size
        self.cat_size_z = cat_size_z
        self.cat_size_u = cat_size_u
        
        # define prior on model parameters for latent variable u
        # informed prior on the Logit-normal distribution
        self.mu_u0_prior = Normal(-2.0,1.0) # latent class = 0
        self.mu_u1_prior = Normal(2.0,1.0) # latent class = 1
        # prior on parameters of Bayesian linear regression for jail time 
        # for latent variable z, we are using different prior for corresponding BLR's coefficients 
        # in terms of each latent class to maintain identifiability in latent classes (low, medium, high)
        self.beta_z0_prior = Normal(-2.0,1.0)
        self.beta_z1_prior = Normal(0.0,1.0)
        self.beta_z2_prior = Normal(2.0,1.0)
        # same prior distribution for rest of the BLR's coefficients  
        self.beta_others_prior = Normal(0.0,1.0)
        # sigma prior is for BLR's sigma parameter
        self.sigma_prior = Gamma(1.0,2.0) # (2,1)
        
        # initialize model parameters for age
        self.mu_a0 = nn.Parameter(torch.randn(1,input_a, requires_grad=True))
        self.mu_a1 = nn.Parameter(torch.randn(1,input_a, requires_grad=True))                                   
        self.logvar_a0 = nn.Parameter(Uniform(-1, 1).sample([1,input_a]), requires_grad=True) # std = torch.exp(0.5*logvar) or Variance = std^2
        self.logvar_a1 = nn.Parameter(Uniform(-1, 1).sample([1,input_a]), requires_grad=True)
        # initialize model parameters for charge degree
        self.mu_c0 = nn.Parameter(torch.randn(1,input_c, requires_grad=True))
        self.mu_c1 = nn.Parameter(torch.randn(1,input_c, requires_grad=True)) 
        self.logvar_c0 = nn.Parameter(Uniform(-1, 1).sample([1,input_c]), requires_grad=True)
        self.logvar_c1 = nn.Parameter(Uniform(-1, 1).sample([1,input_c]), requires_grad=True)
        
        # initialize model parameters of Bayesian linear regression for jail time: t
        self.beta_tz0 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_tz1 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_tz2 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_t0 = nn.Parameter(Normal(0, 1).sample(), requires_grad=True)
        self.beta_tu = nn.Parameter(torch.randn(cat_size_u,1, requires_grad=True))
        self.beta_tc = nn.Parameter(torch.randn(input_c,1, requires_grad=True))         
        self.logvar_t = nn.Parameter(Uniform(-1, 1).sample(), requires_grad=True)
        
        # model network to generate latent variable z: p(z|x_z)
        self.model_z = NeuralNetwork(input_z,cat_size_z,encode_hidden,d_prob,acts)
        # inference network to infer latent variable z: q(z|t)
        self.infer_z = NeuralNetwork(input_t+input_z,cat_size_z,encode_hidden,d_prob,acts)
        # inference network to infer latent variable u: q(u|x_u,t)
        self.infer_u = NeuralNetwork(input_a+input_c+input_t,cat_size_u,encode_hidden,d_prob,acts) 
    
    def reparameterize_latent(self, mu, sigma): 
        eps = torch.randn_like(sigma)
        sample = torch.softmax(mu + (sigma*eps), dim=1) # drawing sample from logistic normal distribution
        return sample
        
    def reconstruct_x_u(self,x_a, x_c):  
        eps_a0 = torch.randn_like(x_a)
        eps_a1 = torch.randn_like(x_a)
        # convert log variance to standard deviation
        sigma_a0 = torch.exp(0.5*self.logvar_a0)
        sigma_a1 = torch.exp(0.5*self.logvar_a1)
        # reparameterization trick
        xHat_a0 = self.mu_a0 + sigma_a0*eps_a0
        xHat_a1 = self.mu_a1 + sigma_a1*eps_a1
        
        eps_c0 = torch.randn_like(x_c)
        eps_c1 = torch.randn_like(x_c)
        # convert log variance to standard deviation
        sigma_c0 = torch.exp(0.5*self.logvar_c0)
        sigma_c1 = torch.exp(0.5*self.logvar_c1)
        # reparameterization trick
        xHat_c0 = self.mu_c0 + sigma_c0*eps_c0
        xHat_c1 = self.mu_c1 + sigma_c1*eps_c1
        
        return xHat_a0, xHat_a1, self.mu_a0, self.mu_a1, sigma_a0, sigma_a1,\
                    xHat_c0, xHat_c1, self.mu_c0, self.mu_c1, sigma_c0, sigma_c1
    
    def parameter_prior_log_prob(self,mu_a0,mu_a1,mu_c0,mu_c1,sigma_a0,sigma_a1,sigma_c0,sigma_c1,beta_t0,beta_tz0,beta_tz1,beta_tz2,beta_tu,beta_tc,sigma_t):
        # log prior for model parameters
        # sorry about this long line of code! I don't have time to simplify this; deadline :(
        log_prior_param = self.mu_u0_prior.log_prob(mu_a0).sum() + self.mu_u1_prior.log_prob(mu_a1).sum() + self.mu_u0_prior.log_prob(mu_c0).sum() \
                            + self.mu_u1_prior.log_prob(mu_c1).sum() + self.sigma_prior.log_prob(sigma_a0).sum() + self.sigma_prior.log_prob(sigma_a1).sum() \
                            + self.sigma_prior.log_prob(sigma_c0).sum() + self.sigma_prior.log_prob(sigma_c1).sum() + self.beta_z0_prior.log_prob(beta_tz0).sum() \
                            + self.beta_z1_prior.log_prob(beta_tz1).sum() + self.beta_z2_prior.log_prob(beta_tz2).sum() + self.beta_others_prior.log_prob(beta_tu).sum() \
                            + self.beta_others_prior.log_prob(beta_tc).sum() + self.beta_others_prior.log_prob(beta_t0) + self.sigma_prior.log_prob(sigma_t)
        return log_prior_param
    
    def forward(self,x_z, x_a, x_c,jailTime,temp):

        # infer latent variable z from q(z|t)
        q_z_pi = self.infer_z(torch.cat((x_z,jailTime), dim=1))
        q_z_y = q_z_pi.view(q_z_pi.size(0), self.latent_size, self.cat_size_z)
        q_z = gumbel_softmax(q_z_y, temp,self.latent_size, self.cat_size_z)
        
        # infer latent variable u from q(u|x_u,t)
        q_u_pi  = self.infer_u(torch.cat((x_a,x_c,jailTime), dim=1))
        q_u_y = q_u_pi.view(q_u_pi.size(0), self.latent_size, self.cat_size_u)
        q_u = gumbel_softmax(q_u_y, temp,self.latent_size, self.cat_size_u)
        
        # reconstruct x_u
        xHat_a0, xHat_a1, mu_a0, mu_a1, sigma_a0, sigma_a1,\
            xHat_c0, xHat_c1, mu_c0, mu_c1, sigma_c0, sigma_c1 = self.reconstruct_x_u(x_a, x_c)
        
        # Bayesian linear regression for jail time
        mu_t0 = self.beta_t0 + q_z @ self.beta_tz0 + q_u @ self.beta_tu + x_c @ self.beta_tc
        mu_t1 = self.beta_t0 + q_z @ self.beta_tz1 + q_u @ self.beta_tu + x_c @ self.beta_tc
        mu_t2 = self.beta_t0 + q_z @ self.beta_tz2 + q_u @ self.beta_tu + x_c @ self.beta_tc

        sigma_t = torch.exp(0.5*self.logvar_t)
        
        eps_t0 =  torch.randn_like(mu_t0) # since we are using single sigma
        tHat0 = mu_t0 + sigma_t*eps_t0 # drawing jail time using reparameterization trick

        eps_t1 =  torch.randn_like(mu_t1) # since we are using single sigma
        tHat1 = mu_t1 + sigma_t*eps_t1 # drawing jail time using reparameterization trick

        eps_t2 =  torch.randn_like(mu_t2) # since we are using single sigma
        tHat2 = mu_t2 + sigma_t*eps_t2 # drawing jail time using reparameterization trick
        
        # compute log_prob for prior on model's parameters
        log_prior_param = self.parameter_prior_log_prob(mu_a0,mu_a1,mu_c0,mu_c1,sigma_a0,sigma_a1,sigma_c0,sigma_c1,self.beta_t0,self.beta_tz0,self.beta_tz1,self.beta_tz2,self.beta_tu,self.beta_tc,sigma_t)
        
        return tHat0, tHat1, tHat2, mu_t0, mu_t1, mu_t2, sigma_t, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, log_prior_param

class prior_network(nn.Module):
    def __init__(self,input_z, latent_size, cat_size_z, encode_hidden, d_prob, acts):
        super(prior_network,self).__init__()
        
        self.input_z = input_z
        self.latent_size = latent_size
        self.cat_size_z = cat_size_z
        
        self.beta_z0_prior = Normal(-1.0,1.0)
        self.beta_z1_prior = Normal(0.0,1.0)
        self.beta_z2_prior = Normal(1.0,1.0)
        # same prior distribution for rest of the BLR's coefficients  
        self.beta_others_prior = Normal(0.0,1.0)
        # sigma prior is for BLR's sigma parameter
        self.sigma_prior = Gamma(1.0,2.0) # (2,1)
        
        # initialize model parameters of Bayesian linear regression for jail time: t
        self.beta_tz0 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_tz1 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_tz2 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_t0 = nn.Parameter(Normal(0, 1).sample(), requires_grad=True)
        self.logvar_t = nn.Parameter(Uniform(-1, 1).sample(), requires_grad=True)
        
        # model network to generate latent variable z: p(z|x_z)
        self.model_z = NeuralNetwork(input_z,cat_size_z,encode_hidden,d_prob,acts)
    
    def reparameterize_latent(self, mu, sigma): 
        eps = torch.randn_like(sigma)
        sample = torch.softmax(mu + (sigma*eps), dim=1) # drawing sample from logistic normal distribution
        return sample
        
    
    def parameter_prior_log_prob(self,beta_t0,beta_tz0,beta_tz1,beta_tz2,sigma_t):
        # log prior for model parameters
        # sorry about this long line of code! I don't have time to simplify this; deadline :(
        log_prior_param = self.beta_z0_prior.log_prob(beta_tz0).sum() + self.beta_z1_prior.log_prob(beta_tz1).sum() + self.beta_z2_prior.log_prob(beta_tz2).sum()\
                    + self.beta_others_prior.log_prob(beta_t0) + self.sigma_prior.log_prob(sigma_t)
        return log_prior_param
    
    def forward(self,x_z,jailTime,temp):
        # generate latent variable z from p(z|x_z)
        p_z_pi = self.model_z(x_z)
        p_z_y = p_z_pi.view(p_z_pi.size(0), self.latent_size, self.cat_size_z)
        p_z = gumbel_softmax(p_z_y, temp,self.latent_size, self.cat_size_z)
        
        # Bayesian linear regression for jail time
        mu_t0 = self.beta_t0 + p_z @ self.beta_tz0 
        mu_t1 = self.beta_t0 + p_z @ self.beta_tz1 
        mu_t2 = self.beta_t0 + p_z @ self.beta_tz2 

        sigma_t = torch.exp(0.5*self.logvar_t)
        
        eps_t0 =  torch.randn_like(mu_t0) # since we are using same sigma
        tHat0 = mu_t0 + sigma_t*eps_t0 # drawing jail time using reparameterization trick

        eps_t1 =  torch.randn_like(mu_t1) # since we are using same sigma
        tHat1 = mu_t1 + sigma_t*eps_t1 # drawing jail time using reparameterization trick

        eps_t2 =  torch.randn_like(mu_t2) # since we are using same sigma
        tHat2 = mu_t2 + sigma_t*eps_t2 # drawing jail time using reparameterization trick
        
        # compute log_prob for prior on model's parameters
        log_prior_param = self.parameter_prior_log_prob(self.beta_t0,self.beta_tz0,self.beta_tz1,self.beta_tz2,sigma_t)
        
        return tHat0, tHat1, tHat2, mu_t0, mu_t1, mu_t2, sigma_t, p_z, log_prior_param

class prior_network_ft(nn.Module):
    def __init__(self,input_z, latent_size, cat_size_z, encode_hidden, d_prob, acts):
        super(prior_network_ft,self).__init__()
        
        self.input_z = input_z
        self.latent_size = latent_size
        self.cat_size_z = cat_size_z
        
        self.beta_z0_prior = Normal(-1.0,1.0)
        self.beta_z1_prior = Normal(0.0,1.0)
        self.beta_z2_prior = Normal(1.0,1.0)
        # same prior distribution for rest of the BLR's coefficients  
        self.beta_others_prior = Normal(0.0,1.0)
        # sigma prior is for BLR's sigma parameter
        self.sigma_prior = Gamma(1.0,2.0) # (2,1)
        
        # initialize model parameters of Bayesian linear regression for jail time: t
        self.beta_tz0 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_tz1 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_tz2 = nn.Parameter(torch.randn(cat_size_z,1, requires_grad=True)) 
        self.beta_t0 = nn.Parameter(Normal(0, 1).sample(), requires_grad=True)
        self.logvar_t = nn.Parameter(Uniform(-1, 1).sample(), requires_grad=True)
        
        # model network to generate latent variable z: p(z|x_z)
        self.model_z = NeuralNetwork(input_z,cat_size_z,encode_hidden,d_prob,acts)
    
    def reparameterize_latent(self, mu, sigma): 
        eps = torch.randn_like(sigma)
        sample = torch.softmax(mu + (sigma*eps), dim=1) # drawing sample from logistic normal distribution
        return sample
    
    def forward(self,x_z,jailTime,temp):
        # generate latent variable z from p(z|x_z)
        p_z_pi = self.model_z(x_z)
        p_z_y = p_z_pi.view(p_z_pi.size(0), self.latent_size, self.cat_size_z)
        p_z = gumbel_softmax(p_z_y, temp,self.latent_size, self.cat_size_z)
        
        return p_z
