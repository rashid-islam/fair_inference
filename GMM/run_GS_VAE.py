import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable

from utilities import load_hhp_data
from models import GaussianVAE
from objective_functions import vae_elbo_function, vae_held_out_logLL, stochasticEDFModel, computeEDFforBatch, df_loss
from evaluations import computeEDF, prob_latent_given_group, prob_latent_given_group_hard, computeDF_hard, groupFairnessMeasures
from sklearn.feature_selection import mutual_info_classif

device = torch.device("cpu")
print(device)

def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

#%% data loading and  pre-processing
# load the train dataset
X, S = load_hhp_data('data/hhp.csv')
# Define all the "intersectional groups"
intersectionalGroups = np.unique(S,axis=0) # all intersecting groups, i.e. black-women, white-man etc  

#%%
# data pre-processing
# scale/normalize train & test data and shuffle train data
X, test_X, S, test_S = train_test_split(X, S, test_size=0.10, stratify=S, random_state=7)

scaler = StandardScaler().fit(X)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X = X.pipe(scale_df, scaler) 
test_X = test_X.pipe(scale_df, scaler)

X = X.values 
test_X = test_X.values 

X, dev_X, S, dev_S = train_test_split(X, S, test_size=0.10,stratify=S, random_state=7)

data_eval_train = X
data_eval_dev = dev_X
data_eval_test = test_X

# convert to torch
X = torch.from_numpy(X)                     
testData = torch.from_numpy(test_X)
devData = torch.from_numpy(dev_X)

testData = Variable(testData.float()).to(device)
devData = Variable(devData.float()).to(device)

#%% deep neural network's hyperparameters
input_size = X.size()[1]
num_epochs = 5

latent_size = 1
cat_size = 3

latent_prior = 1/cat_size

tau_min = 0.5
anneal_rate = 0.00003

# hyper-parameters' grids
encode_hiddens = [[64,64],[64,32],[32,16]]
minibatches = [128,256]
learning_rates = [0.001,0.002,0.005]
drop_probs = [0.1,0.25]
activation_funcs = ['rectify','softplus']
weight_decays = [1e-3,1e-4]

numHypers = len(encode_hiddens)*len(minibatches)*len(learning_rates)*len(drop_probs)*len(activation_funcs)*len(weight_decays)

best_net = None # store the best model into this
best_ll = -1e+20

#%% Report for grid-search experimental results on dev set
coulumn_names = ['ELBO', 'CLL','avg_mi', 'epsilon', 'gamma', 'avg_gamma', 'dp_gender', 'dp_age', 'pRule_gender', 'pRule_age',\
                 'dp_gender_avg', 'dp_age_avg', 'pRule_gender_avg', 'pRule_age_avg','overall_dp','overall_pRule']
Dev_results = np.zeros((numHypers,len(coulumn_names)))
Dev_results = pd.DataFrame(Dev_results, columns=coulumn_names)

#%% Training discrete latent variable model
import sys
sys.stdout=open("results/gsVAE_experimental_report.txt","w")

print(f"Grid search over hyperparameters")
print('\n')
print('\n')

run = 0 # to track n-th model
for encode_hidden in encode_hiddens:
    for minibatch in minibatches: 
        for learning_rate in learning_rates:
                for drop_prob in drop_probs:
                    for activation_func in activation_funcs:
                        for weight_decay in weight_decays:
                            print(f"Model: {run: d}")
                            RANDOM_STATE = np.random.randint(1000)
                            set_random_seed(RANDOM_STATE)

                            gaussian_vae = GaussianVAE(input_size,latent_size,cat_size,encode_hidden,drop_prob,activation_func).to(device)
                            optimizer = optim.Adam(gaussian_vae.parameters(),lr = learning_rate,weight_decay=weight_decay)
                            
                            print(f"Trained with the following hyperparameters:")
                            print('RANDOM_STATE: ', RANDOM_STATE,'hidden layers (nodes/layer): ', encode_hidden, 'learning rate: ', learning_rate, 'Drop out prob: ', drop_prob, 'mini-batch sizes: ', minibatch, 'activation function: ', activation_func, 'weight decay: ', weight_decay)
                            print('\n')
                            
                            # training starts
                            batch_idx = 0
                            tau = 1.0 # initial temp
                            flag = False  # Defining the flag variable to avoid extreme solution of Gumbel softmax
                            gaussian_vae.train()
                            for epoch in range(num_epochs):
                                for batch in range(0,np.int64(np.floor(len(X)/minibatch))*minibatch,minibatch):
                                    trainX_batch = X[batch:(batch+minibatch)]
                                    trainX_batch = Variable(trainX_batch.float()).to(device)
                                    
                                    q_z, log_prob = gaussian_vae(trainX_batch, tau)
                                    
                                    elbo = vae_elbo_function(q_z, log_prob, latent_prior)
                                    loss = - elbo
                                    
                                    if np.isnan(loss.item()) == True:
                                        flag = True
                                        break
                                    
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                                    
                                    batch_idx += 1
                                    if batch_idx%5 == 0:
                                        tau = np.maximum(tau * np.exp(-anneal_rate * batch_idx), tau_min)
                            
                                if flag == True:
                                        break
                            
                            if flag == False:
                                with torch.no_grad():
                                    qz_typ, typ_log_prob = gaussian_vae(devData, tau)
                                Dev_results['ELBO'][run]=vae_elbo_function(qz_typ, typ_log_prob, latent_prior)
                                Dev_results['CLL'][run] = vae_held_out_logLL(qz_typ, typ_log_prob)

                                if qz_typ.device.type == 'cuda':
                                    qz_typ = qz_typ.cpu()
                                qz_typ = np.array(qz_typ)
                                hard_cluster = np.argmax(qz_typ, axis=1)
                                
                                mi_per_feature = mutual_info_classif(data_eval_dev, hard_cluster)
                                Dev_results['avg_mi'][run] = np.mean(mi_per_feature)
                                Dev_results['epsilon'][run], Dev_results['gamma'][run], Dev_results['avg_gamma'][run] = computeEDF(dev_S,qz_typ,intersectionalGroups)
                                Dev_results['dp_gender'][run], Dev_results['dp_age'][run], Dev_results['pRule_gender'][run], Dev_results['pRule_age'][run], Dev_results['dp_gender_avg'][run], Dev_results['dp_age_avg'][run], Dev_results['pRule_gender_avg'][run], Dev_results['pRule_age_avg'][run] = groupFairnessMeasures(dev_S,qz_typ)
                                Dev_results['overall_dp'][run] = (Dev_results['dp_gender_avg'][run]+Dev_results['dp_age_avg'][run])/dev_S.shape[1]
                                Dev_results['overall_pRule'][run] = (Dev_results['pRule_gender_avg'][run]+Dev_results['pRule_age_avg'][run])/dev_S.shape[1]

                                # choose the best model based on a pre-defined condition
                                if Dev_results['CLL'][run]>best_ll and len(np.unique(hard_cluster))>1:
                                    best_ll = Dev_results['CLL'][run]
                                    print('\n')
                                    print(f"Current best model:{run: d}")
                                    print('RANDOM_STATE: ', RANDOM_STATE,'hidden layers (nodes/layer): ', encode_hidden, 'learning rate: ', learning_rate, 'Drop out prob: ', drop_prob, 'mini-batch sizes: ', minibatch, 'activation function: ', activation_func, 'weight decay: ', weight_decay)
                                    print(f"LL on dev set: {Dev_results['CLL'][run]: .3f}")
                                    best_net = gaussian_vae
                                    print('\n')
                                    print('\n')
                            
                            run += 1


#%% Final evaluation on the best model
print('\n')
print('For the selected best model,')
print('\n')
print('Evaluation results on the test set:')
print('\n')

with torch.no_grad():
    qz_typ, typ_log_prob = best_net(testData, tau)

test_elbo_typ=vae_elbo_function(qz_typ, typ_log_prob, latent_prior)
print(f"average ELBO: {test_elbo_typ: .3f}")
test_ll_typ = vae_held_out_logLL(qz_typ, typ_log_prob)
print(f"average CLL: {test_ll_typ: .3f}")

if qz_typ.device.type == 'cuda':
    qz_typ = qz_typ.cpu()

qz_typ = np.array(qz_typ)
hard_cluster = np.argmax(qz_typ, axis=1)

    
db_score = davies_bouldin_score(data_eval_test, hard_cluster)
print(f"davies_bouldin_score: {db_score: .3f}")
ch_score = calinski_harabasz_score(data_eval_test, hard_cluster)
print(f"calinski_harabasz_score: {ch_score: .3f}")

# \epsilon-DF measure
epsilon_typ, gamma_typ, gamma_typ_avg = computeEDF(test_S,qz_typ,intersectionalGroups)
print(f"Differential fairness measures: {epsilon_typ: .3f}")
print(f"Subgroup fairness measures: {gamma_typ: .3f}")
print(f"Avg Subgroup fairness measures: {gamma_typ_avg: .3f}")

# group fairness measure
dp_gender, dp_age, pRule_gender, pRule_age, dp_gender_avg, dp_age_avg, pRule_gender_avg, pRule_age_avg = groupFairnessMeasures(test_S,qz_typ)

print(f"Demographic parity measures (gender): {dp_gender: .3f}")
print(f"Demographic parity measures (age): {dp_age: .3f}")

print(f"p%-rule measures (gender): {pRule_gender: .3f}")
print(f"p%-rule measures (age): {pRule_age: .3f}")

print(f"Avg Demographic parity measures (gender): {dp_gender_avg: .3f}")
print(f"Avg Demographic parity measures (age): {dp_age_avg: .3f}")

print(f"Avg p%-rule measures (gender): {pRule_gender_avg: .3f}")
print(f"Avg p%-rule measures (age): {pRule_age_avg: .3f}")

overall_dp = (dp_gender_avg+dp_age_avg)/test_S.shape[1]
overall_pRule = (pRule_gender_avg+pRule_age_avg)/test_S.shape[1]

print(f"Overall Demographic parity measures: {overall_dp: .3f}")
print(f"Overall p%-rule measures: {overall_pRule: .3f}")
# mutual info
mi_per_feature = mutual_info_classif(data_eval_test, hard_cluster)
print(f"avg mi: {np.mean(mi_per_feature): .3f}")
np.savetxt('results/vae_mi_per_feature_gsVAE.txt',mi_per_feature)

#%% saving model and results
torch.save(best_net.state_dict(), "trained-models/gs_vae")
Dev_results.to_csv('results/Dev_results_gsVAE.csv',index=False)