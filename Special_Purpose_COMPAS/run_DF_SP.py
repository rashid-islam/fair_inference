
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import normalized_mutual_info_score

import matplotlib 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable

from utilities import load_compas_data
from models import SP, prior_network, prior_network_ft
from objective_functions import elbo_function, held_out_logLL, stochasticEDFModel, computeEDFforBatch, df_loss, ml_function
from evaluations import computeEDF, prob_latent_given_group_hard, groupFairnessMeasures

device = torch.device("cpu")
print(device)

def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

#%% data loading and  pre-processing
# load the whole dataset
X_z, X_a, X_c, jail_time, S, compas_pred, y = load_compas_data('data/compas-scores-two-years.csv')
# Define all the "intersectional groups"
intersectionalGroups = np.unique(S,axis=0) # all intersecting groups, i.e. black-women, white-man etc  

X_z = X_z.values 
X_a = X_a.values 
X_c = X_c.values
jail_time = jail_time.values.reshape(-1,1)  
S = S.values
compas_pred = compas_pred.values 
y = y.values

# train-dev-test splits
X_z, test_X_z, X_a, test_X_a, X_c, test_X_c, jail_time, test_jail_time, S, test_S, compas_pred, test_compas_pred, y, test_y =\
         train_test_split(X_z, X_a, X_c, jail_time, S, compas_pred, y, test_size=0.20, stratify=y, random_state=7)

X_z, dev_X_z, X_a, dev_X_a, X_c, dev_X_c, jail_time, dev_jail_time, S, dev_S, compas_pred, dev_compas_pred, y, dev_y =\
         train_test_split(X_z, X_a, X_c, jail_time, S, compas_pred, y, test_size=0.20, stratify=y, random_state=7)

# scale/normalize the continuous observed variables
scaler = StandardScaler().fit(X_z)
X_z = scaler.transform(X_z)
dev_X_z = scaler.transform(dev_X_z)
test_X_z = scaler.transform(test_X_z)

scaler_t = StandardScaler().fit(jail_time)
jail_time = scaler_t.transform(jail_time)
dev_jail_time = scaler_t.transform(dev_jail_time)
test_jail_time = scaler_t.transform(test_jail_time)

# saving the standardization parameters (mean and variance) so that we can use them later, in case we need them for evaluation
import pickle
# save
with open('trained-models/scaler_t.pkl','wb') as f:
    pickle.dump(scaler_t,f)
      
# =============================================================================
# # load
# with open('trained-models/scaler_t.pkl', 'rb') as f:
#     scaler_t_model = pickle.load(f)
# =============================================================================

#%% Converting to torch to use PyTorch framework for training
trainData_X_z = torch.from_numpy(X_z)                     
devData_X_z = torch.from_numpy(dev_X_z)
testData_X_z = torch.from_numpy(test_X_z)

trainData_X_a = torch.from_numpy(X_a)                     
devData_X_a = torch.from_numpy(dev_X_a)
testData_X_a = torch.from_numpy(test_X_a)

trainData_X_c = torch.from_numpy(X_c)                     
devData_X_c = torch.from_numpy(dev_X_c)
testData_X_c = torch.from_numpy(test_X_c)

trainData_jail_time = torch.from_numpy(jail_time)                     
devData_jail_time = torch.from_numpy(dev_jail_time)
testData_jail_time= torch.from_numpy(test_jail_time)

# held-out data conversion for evaluation
devData_X_z = Variable(devData_X_z.float()).to(device)
testData_X_z = Variable(testData_X_z.float()).to(device)

devData_X_a = Variable(devData_X_a.float()).to(device)
testData_X_a = Variable(testData_X_a.float()).to(device)

devData_X_c = Variable(devData_X_c.float()).to(device)
testData_X_c = Variable(testData_X_c.float()).to(device)

devData_jail_time = Variable(devData_jail_time.float()).to(device)
testData_jail_time = Variable(testData_jail_time.float()).to(device)

#%% deep neural network's hyperparameters
input_z = trainData_X_z.size()[1]
input_a = trainData_X_a.size()[1]
input_c = trainData_X_c.size()[1]
input_t = trainData_jail_time.size()[1]

num_epochs = 50
cat_size_z = 3 # 3 latent class, risk of crime: low, medium, high
cat_size_u = 2 # 2 latent class, systems of oppression: no, yes

u_prior = 1/cat_size_u

latent_size = 1

tau_min = 0.5
anneal_rate = 0.00003
#%% hyper-parameters selected in terms of performance on dev set for vanilla model
encode_hidden = [64,32]
minibatch = 128
learning_rate = 0.005
drop_prob = 0.1
activation_func = 'rectify'
weight_decay = 1e-4

RANDOM_STATE =  1 
set_random_seed(RANDOM_STATE)

#%% pre-train the prior network like Vanilla-SP to address posterior collapse is not required for DF-SP model 

#%% Train the special purpose model, while freezing the prior network     
import sys
sys.stdout=open("results/dfSP_experimental_report.txt","w")

lamda_grid = [0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 75.0, 100.0]

slack = 0.15 # small slack tolerance (1.5% degradation) in performance (higher comparing to other model, since LL on only Jail time is lower comparing to other models where we considered all observed)
best_vanilla_ll = -1311.138 # log-likelihood of the best vanilla model on dev set
threshold_ll = best_vanilla_ll + best_vanilla_ll*slack # to pick allowable solutions
best_fairness = 1.234 # initialize with fairness of the best vanilla model for dev set

# fairness intervention specific hyperparameters
stepSize = 0.09 
epsilonBase = torch.tensor(0.0,device=device)

numHypers = len(lamda_grid)
# Report for grid-search experimental results on dev set
coulumn_names = ['ELBO', 'MLL','CLL', 'LL', 'y_mi', 'c_mi', 'r2', 'mse', 'mae', 'epsilon', 'gamma','dp_race', 'dp_gender', 'pRule_race', 'pRule_gender']
Dev_results = np.zeros((numHypers,len(coulumn_names)))
Dev_results = pd.DataFrame(Dev_results, columns=coulumn_names)

#%% training starts for fair model
print(f"Grid search over hyperparameters")
print('\n')
print('\n')

run = 0 # to track n-th model
for ld in lamda_grid:
    print(f"Model: {run: d}")
    lamda = torch.tensor(ld,device=device)
    
    set_random_seed(RANDOM_STATE)
    fair_SP = SP(input_z, input_a, input_c, input_t, latent_size, cat_size_z, cat_size_u, encode_hidden, drop_prob, activation_func).to(device)
    p_net_ft = prior_network_ft(input_z, latent_size, cat_size_z, encode_hidden, drop_prob, activation_func).to(device)
    
    optimizer = optim.Adam(list(p_net_ft.parameters())+list(fair_SP.parameters()),lr = learning_rate, weight_decay=weight_decay)
    vb_EDFModel = stochasticEDFModel(len(intersectionalGroups),cat_size_z,len(trainData_jail_time),minibatch)
    
    print(f"Trained with the following hyperparameters:")
    print('lambda: ', ld)
    print('\n')
    
    # training starts
    batch_idx = 0
    tau = 1.0 # initial temp
    flag = False  
    fair_SP.train()
    for epoch in range(num_epochs):
        for batch in range(0,np.int64(np.floor(len(trainData_jail_time)/minibatch))*minibatch,minibatch):
            
            trainS_batch = S[batch:(batch+minibatch)]
            
            trainT_batch = trainData_jail_time[batch:(batch+minibatch)]
            trainT_batch = Variable(trainT_batch.float()).to(device)
    
            trainXZ_batch = trainData_X_z[batch:(batch+minibatch)]
            trainXZ_batch = Variable(trainXZ_batch.float()).to(device)
    
            trainXA_batch = trainData_X_a[batch:(batch+minibatch)]
            trainXA_batch = Variable(trainXA_batch.float()).to(device)
            
            trainXC_batch = trainData_X_c[batch:(batch+minibatch)]
            trainXC_batch = Variable(trainXC_batch.float()).to(device)
            
            vb_EDFModel.countClass_hat.detach_()
            vb_EDFModel.countTotal_hat.detach_()
            
            #with torch.no_grad():
            p_z = p_net_ft(trainXZ_batch,trainT_batch,tau)
                
            tHat0, tHat1, tHat2, mu_t0, mu_t1, mu_t2, sigma_t, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, \
                         log_prior_param  = fair_SP(trainXZ_batch, trainXA_batch, trainXC_batch,trainT_batch,tau)
            
            elbo = elbo_function(trainXA_batch, trainXC_batch, trainT_batch, mu_t0, mu_t1, mu_t2, sigma_t, p_z, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, log_prior_param, u_prior, fair_SP)
            loss = - elbo
            
            if np.isnan(loss.item()) == True:
                flag = True
                break
            
            # update EDF model 
            countClass, countTotal = computeEDFforBatch(trainS_batch,intersectionalGroups,q_z)
            vb_EDFModel(stepSize,countClass, countTotal,len(trainData_jail_time),minibatch)
            
            # fairness constraint 
            epsilon_loss = df_loss(epsilonBase,vb_EDFModel)            
            tot_loss = loss + lamda*epsilon_loss
                                        
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
        
            batch_idx += 1
            if batch_idx%5 == 0:
                tau = np.maximum(tau * np.exp(-anneal_rate * batch_idx), tau_min)
        
        if flag == True:
            break
                                    
    if flag == False:
        fair_SP.eval()
        with torch.no_grad():
            p_z = p_net_ft(devData_X_z,devData_jail_time,tau)
            tHat0, tHat1, tHat2, mu_t0, mu_t1, mu_t2, sigma_t, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, \
                log_prior_param = fair_SP(devData_X_z, devData_X_a, devData_X_c, devData_jail_time,tau)
        
        Dev_results['ELBO'][run] = elbo_function(devData_X_a, devData_X_c, devData_jail_time, mu_t0, mu_t1, mu_t2, sigma_t, p_z, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, log_prior_param, u_prior, fair_SP)
        
        Dev_results['LL'][run], Dev_results['MLL'][run], Dev_results['CLL'][run] = held_out_logLL(devData_X_a, devData_X_c, devData_jail_time, q_z, mu_t0, mu_t1, mu_t2, sigma_t, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1)
        
        if q_z.device.type == 'cuda':
            q_z = q_z.cpu()
            q_u = q_u.cpu()
            tHat0 = tHat0.cpu()
            tHat1 = tHat1.cpu()
            tHat2 = tHat2.cpu()
            mu_t0 = mu_t0.cpu()
            mu_t1 = mu_t1.cpu()
            mu_t2 = mu_t2.cpu()
        
        q_z = np.array(q_z)
        hard_cluster_z = np.argmax(q_z, axis=1)
        
        # mutual info
        Dev_results['y_mi'][run] = normalized_mutual_info_score(dev_y,hard_cluster_z)        
        Dev_results['c_mi'][run] = normalized_mutual_info_score(dev_compas_pred,hard_cluster_z)
        
        tHat = torch.zeros_like(tHat0)
        mu_t = torch.zeros_like(mu_t0)
        for i in range(len(hard_cluster_z)):
            if hard_cluster_z[i] == 0:
                tHat[i] = tHat0[i]
                mu_t[i] = mu_t0[i]
            elif hard_cluster_z[i] == 1:
                tHat[i] = tHat1[i]
                mu_t[i] = mu_t1[i]
            else:
                tHat[i] = tHat2[i]
                mu_t[i] = mu_t2[i]
                
        tHat = np.array(tHat)
        
        Dev_results['r2'][run] = r2_score(dev_jail_time, tHat)
        Dev_results['mse'][run] = mean_squared_error(dev_jail_time, tHat)
        Dev_results['mae'][run] = mean_absolute_error(dev_jail_time, tHat)
        
        # \epsilon-DF measure
        Dev_results['epsilon'][run], Dev_results['gamma'][run], gamma_typ_avg = computeEDF(dev_S,q_z,intersectionalGroups)
        # group fairness measure
        Dev_results['dp_race'][run], Dev_results['dp_gender'][run], Dev_results['pRule_race'][run], Dev_results['pRule_gender'][run], dp_race_avg, dp_gender_avg, pRule_race_avg, pRule_gender_avg = groupFairnessMeasures(dev_S,q_z)
        # choose the best model based on a pre-defined condition
        if Dev_results['LL'][run]>=threshold_ll and Dev_results['epsilon'][run]<best_fairness:
            best_fairness = Dev_results['epsilon'][run]
            print('\n')
            print(f"Current best model:{run: d}")
            print('lambda: ', ld)
            print(f"LL on dev set: {Dev_results['LL'][run]: .3f}")
            print(f"DF on dev set: {Dev_results['epsilon'][run]: .3f}")
            best_net = fair_SP
            best_pnet = p_net_ft
            print('\n')
            print('\n')
    
    run += 1
#%% Final evaluation on the best model
print('\n')
print('For the selected best model,')
print('\n')

Dev_results.to_csv('results/Dev_results_dfSP.csv',index=False)

print('Evaluation results on the test set:')
print('\n')

best_net.eval()
best_pnet.eval()
with torch.no_grad():
    p_z = best_pnet(testData_X_z,testData_jail_time,tau)
    tHat0, tHat1, tHat2, mu_t0, mu_t1, mu_t2, sigma_t, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, \
            log_prior_param = best_net(testData_X_z, testData_X_a, testData_X_c, testData_jail_time,tau)

elbo = elbo_function(testData_X_a, testData_X_c, testData_jail_time, mu_t0, mu_t1, mu_t2, sigma_t, p_z, q_z, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1, \
            log_prior_param, u_prior, best_net)

print(f"average ELBO: {elbo: .3f}")

logLL_jaitTime, marginalLL_model, conditionalLL_model = held_out_logLL(testData_X_a, testData_X_c, testData_jail_time, q_z, mu_t0, mu_t1, mu_t2, sigma_t, q_u, xHat_a0, xHat_a1, xHat_c0, xHat_c1)
print(f"marginalLL_model: {marginalLL_model: .3f}")
print(f"conditionalLL_model: {conditionalLL_model: .3f}")
print(f"logLL_jaitTime: {logLL_jaitTime: .3f}")

if q_z.device.type == 'cuda':
    q_z = q_z.cpu()
    q_u = q_u.cpu()
    tHat0 = tHat0.cpu()
    tHat1 = tHat1.cpu()
    tHat2 = tHat2.cpu()
    mu_t0 = mu_t0.cpu()
    mu_t1 = mu_t1.cpu()
    mu_t2 = mu_t2.cpu()

q_z = np.array(q_z)
hard_cluster_z = np.argmax(q_z, axis=1)

q_u = np.array(q_u)
hard_cluster_u = np.argmax(q_u, axis=1)

# mutual info
mi_score_z = normalized_mutual_info_score(test_y,hard_cluster_z)
print(f"avg mi with actual recidivism: {mi_score_z: .3f}")

mi_score_z = normalized_mutual_info_score(test_compas_pred,hard_cluster_z)
print(f"avg mi with COMPAS's categorical score: {mi_score_z: .3f}")

tHat = torch.zeros_like(tHat0)
mu_t = torch.zeros_like(mu_t0)
for i in range(len(hard_cluster_z)):
    if hard_cluster_z[i] == 0:
        tHat[i] = tHat0[i]
        mu_t[i] = mu_t0[i]
    elif hard_cluster_z[i] == 1:
        tHat[i] = tHat1[i]
        mu_t[i] = mu_t1[i]
    else:
        tHat[i] = tHat2[i]
        mu_t[i] = mu_t2[i]
        
tHat = np.array(tHat)

r2 = r2_score(test_jail_time, tHat)
mse = mean_squared_error(test_jail_time, tHat)
mae = mean_absolute_error(test_jail_time, tHat)
print(f"R2 regression score: {r2: .3f}")
print(f"mean squared error: {mse: .3f}")
print(f"mean absolute error: {mae: .3f}")

print('\n')
print('Fairness measure for z:')
print('\n')

# \epsilon-DF measure
epsilon_typ, gamma_typ, gamma_typ_avg = computeEDF(test_S,q_z,intersectionalGroups)
print(f"Differential fairness measures: {epsilon_typ: .3f}")
print(f"Subgroup fairness measures: {gamma_typ: .3f}")
print(f"Avg Subgroup fairness measures: {gamma_typ_avg: .3f}")

# group fairness measure
dp_race, dp_gender, pRule_race, pRule_gender, dp_race_avg, dp_gender_avg, pRule_race_avg, pRule_gender_avg = groupFairnessMeasures(test_S,q_z)

print(f"Demographic parity measures (race): {dp_race: .3f}")
print(f"Demographic parity measures (gender): {dp_gender: .3f}")

print(f"p%-rule measures (race): {pRule_race: .3f}")
print(f"p%-rule measures (gender): {pRule_gender: .3f}")

print(f"Avg Demographic parity measures (race): {dp_race_avg: .3f}")
print(f"Avg Demographic parity measures (gender): {dp_gender_avg: .3f}")

print(f"Avg p%-rule measures (race): {pRule_race_avg: .3f}")
print(f"Avg p%-rule measures (gender): {pRule_gender_avg: .3f}")

overall_dp = (dp_race_avg+dp_gender_avg)/test_S.shape[1]
overall_pRule = (pRule_race_avg+pRule_gender_avg)/test_S.shape[1]

print(f"Overall Demographic parity measures: {overall_dp: .3f}")
print(f"Overall p%-rule measures: {overall_pRule: .3f}")

print('\n')
print('Fairness measure for u:')
print('\n')

# \epsilon-DF measure
epsilon_typ, gamma_typ, gamma_typ_avg = computeEDF(test_S,q_u,intersectionalGroups)
print(f"Differential fairness measures: {epsilon_typ: .3f}")
print(f"Subgroup fairness measures: {gamma_typ: .3f}")
print(f"Avg Subgroup fairness measures: {gamma_typ_avg: .3f}")

# group fairness measure
dp_race, dp_gender, pRule_race, pRule_gender, dp_race_avg, dp_gender_avg, pRule_race_avg, pRule_gender_avg = groupFairnessMeasures(test_S,q_u)

print(f"Demographic parity measures (race): {dp_race: .3f}")
print(f"Demographic parity measures (gender): {dp_gender: .3f}")

print(f"p%-rule measures (race): {pRule_race: .3f}")
print(f"p%-rule measures (gender): {pRule_gender: .3f}")

print(f"Avg Demographic parity measures (race): {dp_race_avg: .3f}")
print(f"Avg Demographic parity measures (gender): {dp_gender_avg: .3f}")

print(f"Avg p%-rule measures (race): {pRule_race_avg: .3f}")
print(f"Avg p%-rule measures (gender): {pRule_gender_avg: .3f}")

overall_dp = (dp_race_avg+dp_gender_avg)/test_S.shape[1]
overall_pRule = (pRule_race_avg+pRule_gender_avg)/test_S.shape[1]

print(f"Overall Demographic parity measures: {overall_dp: .3f}")
print(f"Overall p%-rule measures: {overall_pRule: .3f}")

z_perGroup = prob_latent_given_group_hard(test_S,intersectionalGroups,q_z)
np.savetxt('results/df_qz_perGroup_dfSP.txt',z_perGroup)

u_perGroup = prob_latent_given_group_hard(test_S,intersectionalGroups,q_u)
np.savetxt('results/df_qu_perGroup_dfSP.txt',u_perGroup)

tHat_inv = scaler_t.inverse_transform(tHat)
tHat_perGroup = prob_latent_given_group_hard(test_S,intersectionalGroups,tHat_inv)
np.savetxt('results/tHat_perGroup_dfSP.txt',tHat_perGroup)

#%% saving model and results
torch.save(best_net.state_dict(), "trained-models/df_sp")