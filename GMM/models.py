import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions import MultivariateNormal, Normal
from Wishart import InverseWishart

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

#%% mixture model
class GMM(nn.Module):
    def __init__(self,input_size,latent_size, categorical_size, encode_hidden,d_prob,acts,mu_prior_info):
        super(GMM,self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.categorical_size = categorical_size
        
        # define prior on model parameters
        alpha_Mu = ((torch.from_numpy(mu_prior_info)).to(device)).float() 
        alpha_Sigma = torch.eye(input_size, device=device)
        alpha_Sigma = alpha_Sigma.repeat(latent_size*categorical_size, 1, 1)
        alpha_df = torch.tensor(input_size+2, device=device)
        alpha_L = torch.linalg.cholesky(alpha_Sigma/alpha_df) # a lower-triangular matrix with positive-valued diagonal entries via Cholesky decomposition of the covariance
        
        self.mu_prior_dist = MultivariateNormal(alpha_Mu,scale_tril=alpha_L)
        self.sigma_prior_dist = InverseWishart(df=alpha_df, scale_tril=alpha_L)
        # initialize model parameters
        mu_prior_info = ((torch.from_numpy(mu_prior_info)).to(device)).float() 
        self.mu = torch.nn.Parameter(mu_prior_info, requires_grad=True)
        
        cov_factor = torch.eye(input_size)/input_size
        self.cov_factor = torch.nn.Parameter(cov_factor.repeat(latent_size*categorical_size, 1, 1),requires_grad=True)

        self.encoder_net = NeuralNetwork(input_size,latent_size*categorical_size,encode_hidden,d_prob,acts)
        
    def encoder(self,x):
        out = self.encoder_net(x)
        return out
    
    def converter(self,x):   
        # map input data by repeating for easier computation with multi-dimensional model parameters
        out=x.unsqueeze(1)
        out=out.repeat(1,self.latent_size*self.categorical_size,1)
        return out
    
    def forward(self,x, temp):
        q = self.encoder(x)
        q_y = q.view(q.size(0), self.latent_size, self.categorical_size)
        z = gumbel_softmax(q_y, temp,self.latent_size, self.categorical_size)
        x_mapped = self.converter(x)
        # covariance matrix estimation
        Sigma = self.cov_factor @ self.cov_factor.transpose(-2, -1) + torch.eye(x.shape[1],device=device).repeat(self.latent_size*self.categorical_size,1,1)
        # convert covariance to a lower triangular matrix for efficient computation
        L_Sigma = torch.linalg.cholesky(Sigma)
        gmm_dist = MultivariateNormal(self.mu, scale_tril=L_Sigma)
        log_density = gmm_dist.log_prob(x_mapped)
        return z, self.mu, Sigma, log_density #  torch.softmax(q_y, dim=-1).reshape(*q.size())

#%% Categorical VAE using Gumbel-Softmax along with Gaussian MLP as decoder
class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, x):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(x - loc, 2) / (2 * var)
    
class  GaussianVAE(nn.Module):
    def __init__(self,input_size,latent_size, categorical_size,encode_hidden,d_prob, acts):
        super(GaussianVAE,self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.categorical_size = categorical_size
        self.log_prob = NormalLogProb() 
        
        # initialize model parameters
        self.encoder_net = NeuralNetwork(input_size,latent_size*categorical_size,encode_hidden,d_prob,acts)
        self.decoder_net = NeuralNetwork(latent_size*categorical_size,input_size*2,list(reversed(encode_hidden)),d_prob,acts)
        
    def encoder(self,x):
        out = self.encoder_net(x)
        return out
    
    def decoder(self,x):  
        out = self.decoder_net(x)
        out = out.view(-1, 2, self.input_size)
        # get `mu` and `log_var`
        mu = out[:, 0, :] 
        logvar = out[:, 1, :]
        return mu, logvar
    
    def forward(self,x, temp):
        q = self.encoder(x)
        q_y = q.view(q.size(0), self.latent_size, self.categorical_size)
        z = gumbel_softmax(q_y, temp,self.latent_size, self.categorical_size)
        mu, logvar = self.decoder(z)
        # convert log variance to standard deviation
        sigma =  torch.exp(0.5*logvar) # self.softplus(logvar)
        #gaussian_dist = Normal(mu, sigma)
        #log_density = gaussian_dist.log_prob(x)
        log_density = self.log_prob(mu, sigma, x)
        return z, log_density

class  GaussianVFAE(nn.Module):
    def __init__(self,input_size,latent_size, categorical_size,group_size,encode_hidden,d_prob, acts):
        super(GaussianVFAE,self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.categorical_size = categorical_size
        self.log_prob = NormalLogProb()
        
        
        # initialize model parameters
        self.encoder_net = NeuralNetwork(input_size,latent_size*categorical_size,encode_hidden,d_prob,acts)
        self.decoder_net = NeuralNetwork((latent_size*categorical_size)+group_size,input_size*2,list(reversed(encode_hidden)),d_prob,acts)
        
    def encoder(self,x):
        out = self.encoder_net(x)
        return out
    
    def decoder(self,x):  
        out = self.decoder_net(x)
        out = out.view(-1, 2, self.input_size)
        # get `mu` and `log_var`
        mu = out[:, 0, :] 
        logvar = out[:, 1, :]
        return mu, logvar
    
    def forward(self,x, s, temp):
        q = self.encoder(x)
        q_y = q.view(q.size(0), self.latent_size, self.categorical_size)
        z = gumbel_softmax(q_y, temp,self.latent_size, self.categorical_size)
        mu, logvar = self.decoder(torch.cat((z, s), dim=1))
        # convert log variance to standard deviation
        sigma =  torch.exp(0.5*logvar) 
        log_density = self.log_prob(mu, sigma, x)
        return z, log_density
