import torch
import torch.nn as nn
import numpy as np

device = torch.device("cpu")
#%% ELBO function
def log_likelihood(x, y_z0, y_z1, observ_dict, qz):
    log_like_qz0 = 0.0
    log_like_qz1 = 0.0
    # log-likelihood for generalized Bernoulli distribution
    for obs in observ_dict:
        log_like_qz0 += torch.sum(qz[:,0].view(-1,1)*(x[:,obs]*torch.log(torch.softmax(y_z0[:,obs],1))), dim=-1)
        log_like_qz1 += torch.sum(qz[:,1].view(-1,1)*(x[:,obs]*torch.log(torch.softmax(y_z1[:,obs],1))), dim=-1) 
    
    return log_like_qz0 + log_like_qz1
    

def elbo_function(x, beta_z0, beta_z1, mu_z0, mu_z1, sigma_z0, sigma_z1, model, observ_dict, qz, prior, log_prior_param, eps=1e-20):
    
    # log-likelihood
    log_like = log_likelihood(x, beta_z0, beta_z1, observ_dict, qz)

    # kl divergence
    KLD = torch.sum(qz * (torch.log(qz/prior + eps)), dim=-1)

    elbo = (log_like - KLD + log_prior_param ).mean()

    return elbo

def vae_elbo_function(x, out_x, observ_dict, qz, prior, eps=1e-20):
    # log-likelihood    
    log_like = 0.0
    for obs in observ_dict:
        log_like += torch.sum(x[:,obs]*torch.log(torch.softmax(out_x[:,obs],1)), dim=-1)
    
    # kl divergence
    KLD = torch.sum(qz * (torch.log(qz/prior + eps)), dim=-1)

    elbo = (log_like - KLD).mean()    

    return elbo

#%% loh-likelihood on held-out data
def held_out_logLL(x, y_z0, y_z1, observ_dict, qz):
    
    log_like_qz0 = 0.0
    log_like_qz1 = 0.0
    # log-likelihood for generalized Bernoulli distribution
    for obs in observ_dict:
        log_like_qz0 += torch.sum(qz[:,0].view(-1,1)*(x[:,obs]*torch.log(torch.softmax(y_z0[:,obs],1))), dim=-1)
        log_like_qz1 += torch.sum(qz[:,1].view(-1,1)*(x[:,obs]*torch.log(torch.softmax(y_z1[:,obs],1))), dim=-1)
    marginalLL = (log_like_qz0 + log_like_qz1).sum()
    
    hard_z = torch.argmax(qz,1)
    conditionalLL = 0.0
    for i in range(len(hard_z)):
        if hard_z[i] == 0:
            for obs in observ_dict:
                conditionalLL += torch.sum(x[i,obs]*torch.log(torch.softmax(y_z0[i,obs],0)))
        elif hard_z[i] == 1:
            for obs in observ_dict:
                conditionalLL += torch.sum(x[i,obs]*torch.log(torch.softmax(y_z1[i,obs],0)))
    return marginalLL, conditionalLL

def vae_held_out_logLL(x, out_x, observ_dict, qz):
    # log-likelihood    
    log_like = 0.0
    for obs in observ_dict:
        log_like += torch.sum(x[:,obs]*torch.log(torch.softmax(out_x[:,obs],1)), dim=-1)

    return log_like.sum()
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
