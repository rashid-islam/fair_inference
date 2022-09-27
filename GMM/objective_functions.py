import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal   #, Gamma
import numpy as np

from Wishart import InverseWishart
device = torch.device("cpu")
#%% ELBO function
def elbo_function(qz, param_Mu, param_Sigma, log_prob, prior_z, model, eps=1e-20):
    
    log_prob_mu0 = model.mu_prior_dist.log_prob(param_Mu).sum()
    log_prob_sigma0 = model.sigma_prior_dist.log_prob(param_Sigma).sum()
    param_prior = log_prob_mu0 + log_prob_sigma0
    
    # log-likelihood
    log_like = qz[:,0]*log_prob[:,0] + qz[:,1]*log_prob[:,1] + qz[:,2]*log_prob[:,2]                                  

    # kl divergence
    KLD = torch.sum(qz * (torch.log(qz/prior_z + eps)), dim=-1)

    elbo = (log_like - KLD + param_prior).mean()

    return elbo

def vae_elbo_function(qz, log_prob, prior_z, eps=1e-20):
    # log-likelihood
    log_like = torch.sum(log_prob, dim = -1)

    # kl divergence
    KLD = torch.sum(qz * (torch.log(qz/prior_z + eps)), dim=-1)

    elbo = (log_like - KLD).mean()    

    return elbo

#%% loh-likelihood on held-out data
def held_out_logLL(qz, log_prob, eps=1e-20):
    # log-likelihood
    marginal_log_like = (qz[:,0]*log_prob[:,0] + qz[:,1]*log_prob[:,1] + qz[:,2]*log_prob[:,2]).sum() 
    
    conditional_log_like = 0.0
    hard_z = torch.argmax(qz,1)
    for i in range(len(hard_z)):
        if hard_z[i] == 0:
            conditional_log_like += log_prob[i,0]
        elif hard_z[i] == 1:
            conditional_log_like += log_prob[i,1]
        elif hard_z[i] == 2:
            conditional_log_like += log_prob[i,2]
    
    return marginal_log_like, conditional_log_like

def vae_held_out_logLL(qz, log_prob, eps=1e-20):
    # log-likelihood
    log_like = torch.sum(log_prob, dim = -1)
    
    return log_like.sum()

#%% Fairness intervations
# compute corresponding counts for each intersection group in a mini-batch
def computeEDFforBatch(protectedAttributes,intersectGroups,latent_distribution):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S) 
    # can be maintained correctly among different batches   
    # compute counts for each intersectional group
    countsClassOne = torch.zeros((len(intersectGroups),latent_distribution.shape[1]),dtype=torch.float,device=device)
    countsTotal = torch.zeros((len(intersectGroups),latent_distribution.shape[1]),dtype=torch.float,device=device)
    for i in range(len(latent_distribution)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index,:] = countsTotal[index,:] + 1.0
        countsClassOne[index,:] = countsClassOne[index,:] + latent_distribution[i,:]
    return countsClassOne, countsTotal

# update count model
class stochasticEDFModel(nn.Module):
    # Source: directly taken from deepSurv implementation
    def __init__(self,no_of_groups,latent_size,N,batch_size):
        super(stochasticEDFModel, self).__init__()
        self.countClass_hat = torch.ones((no_of_groups,latent_size),device=device)
        self.countTotal_hat = torch.ones((no_of_groups,latent_size),device=device)
        
        self.countClass_hat = self.countClass_hat*(N/(batch_size*no_of_groups)) 
        self.countTotal_hat = self.countTotal_hat*(N/batch_size) 
        
    def forward(self,rho,countClass_batch,countTotal_batch,N,batch_size):
        self.countClass_hat = (1-rho)*self.countClass_hat + rho*(N/batch_size)*countClass_batch
        self.countTotal_hat = (1-rho)*self.countTotal_hat + rho*(N/batch_size)*countTotal_batch

        
#%%
# \epsilon-DF measurement to form DF-based fairness penalty
def df_train(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(len(probabilitiesOfPositive),dtype=torch.float,device=device)
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0,device=device) # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                for y in range(probabilitiesOfPositive.shape[1]):
                    epsilon = torch.max(epsilon,torch.abs(torch.log(probabilitiesOfPositive[i,y])-torch.log(probabilitiesOfPositive[j,y]))) # ratio of probabilities of positive outcome
                    #epsilon = torch.max(epsilon,torch.abs((torch.log(1-probabilitiesOfPositive[i,y]))-(torch.log(1-probabilitiesOfPositive[j,y])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = torch.max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

#%% \epsilon-DF fairness penalty
def df_loss(base_fairness,stochasticModel):
    numClasses = torch.tensor(stochasticModel.countClass_hat.shape[1],device=device)
    concentrationParameter = torch.tensor(1.0,device=device)
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0,device=device) 
    
    theta = (stochasticModel.countClass_hat + dirichletAlpha) /(stochasticModel.countTotal_hat + concentrationParameter)
    #theta = theta/sum(theta)
    epsilon_latentDist = df_train(theta)
    return torch.max(zeroTerm, (epsilon_latentDist-base_fairness))
