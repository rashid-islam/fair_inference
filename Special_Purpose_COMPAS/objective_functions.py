import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Normal

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% ELBO function    
def NormalLogProb(loc, scale, x):
    var = torch.pow(scale, 2)
    return (-0.5 * torch.log(2 * np.pi * var) - torch.pow(x - loc, 2) / (2 * var)).sum(dim=-1)
    
def elbo_function(x_a, x_c, jailTime, mu_t0, mu_t1, mu_t2, sigma_t, p_z, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, \
                  log_prior_param, u_prior, model, eps=1e-20):
    
    # log-likelihood
    log_like_a = (q_u[:,0]*torch.sum((x_a*torch.log(torch.softmax(xHat_a0,1))),dim=-1))\
                    + (q_u[:,1]*torch.sum((x_a*torch.log(torch.softmax(xHat_a1,1))),dim=-1))

    log_like_c = (q_u[:,0]*torch.sum((x_c*torch.log(torch.softmax(xHat_c0,1))),dim=-1)) \
                    + (q_u[:,1]*torch.sum((x_c*torch.log(torch.softmax(xHat_c1,1))),dim=-1))
    
    log_like_t0 = NormalLogProb(mu_t0,sigma_t,jailTime)
    log_like_t1 = NormalLogProb(mu_t1,sigma_t,jailTime)
    log_like_t2 = NormalLogProb(mu_t2,sigma_t,jailTime)
    
    log_like_t = q_z[:,0]*log_like_t0 + q_z[:,1]*log_like_t1 + q_z[:,2]*log_like_t2
    

    # kl divergence    
    KLD_z = torch.sum(q_z * (torch.log(q_z/p_z + eps)), dim=-1)
    KLD_u = torch.sum(q_u * (torch.log(q_u/u_prior + eps)), dim=-1)
    
    # elbo for complete data
    elbo = (log_like_t + log_like_a + log_like_c - KLD_z - KLD_u + log_prior_param).mean()

    return elbo

def ml_function(jailTime, mu_t0, mu_t1, mu_t2, sigma_t, p_z, p_log_prior, model):
    
    # log-likelihood
    log_like_t0 = NormalLogProb(mu_t0,sigma_t,jailTime)
    log_like_t1 = NormalLogProb(mu_t1,sigma_t,jailTime)
    log_like_t2 = NormalLogProb(mu_t2,sigma_t,jailTime)
    
    log_like_t = p_z[:,0]*log_like_t0 + p_z[:,1]*log_like_t1 + p_z[:,2]*log_like_t2
    
    # elbo for complete data
    ml = (log_like_t + p_log_prior).mean()

    return ml 

#%% marginal and conditional loh-likelihood on held-out data
def held_out_logLL(x_a, x_c, jailTime, q_z, mu_t0, mu_t1, mu_t2, sigma_t, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1):
    
    # log-likelihood
    log_like_a = (q_u[:,0]*torch.sum((x_a*torch.log(torch.softmax(xHat_a0,1))),dim=-1)) \
                    + (q_u[:,1]*torch.sum((x_a*torch.log(torch.softmax(xHat_a1,1))),dim=-1))

    log_like_c = (q_u[:,0]*torch.sum((x_c*torch.log(torch.softmax(xHat_c0,1))),dim=-1))\
                    + (q_u[:,1]*torch.sum((x_c*torch.log(torch.softmax(xHat_c1,1))),dim=-1))
    
    log_like_t0 = NormalLogProb(mu_t0,sigma_t,jailTime)
    log_like_t1 = NormalLogProb(mu_t1,sigma_t,jailTime)
    log_like_t2 = NormalLogProb(mu_t2,sigma_t,jailTime)
    
    log_like_t = q_z[:,0]*log_like_t0 + q_z[:,1]*log_like_t1 + q_z[:,2]*log_like_t2
    
    marginalLL_model = (log_like_t+log_like_a+log_like_c).sum()
    
    hard_z = torch.argmax(q_z,1)
    logLL_jailTime = torch.zeros_like(log_like_t0)
    for i in range(len(hard_z)):
        if hard_z[i] == 0:
            logLL_jailTime[i] = log_like_t0[i]
        elif hard_z[i] == 1:
            logLL_jailTime[i] = log_like_t1[i]
        else:
            logLL_jailTime[i] = log_like_t2[i]
        
    logLL_jailTime = logLL_jailTime.sum()
    
    hard_u = torch.argmax(q_u,1)
    conditionalLL = 0.0
    for i in range(len(hard_u)):
        if hard_u[i] == 0:
            conditionalLL += torch.sum((x_a[i]*torch.log(torch.softmax(xHat_a0[i],0))))+torch.sum((x_c[i]*torch.log(torch.softmax(xHat_c0[i],0))))
        elif hard_u[i] == 1:
            conditionalLL += torch.sum((x_a[i]*torch.log(torch.softmax(xHat_a1[i],0))))+torch.sum((x_c[i]*torch.log(torch.softmax(xHat_c1[i],0))))
    
    conditionalLL_model = conditionalLL + logLL_jailTime
    
    return logLL_jailTime, marginalLL_model, conditionalLL_model

# =============================================================================
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

