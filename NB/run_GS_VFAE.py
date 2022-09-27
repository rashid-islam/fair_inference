
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

from utilities import load_census_data
from models import BernoulliVFAE
from objective_functions import vae_elbo_function, vae_held_out_logLL, stochasticEDFModel, computeEDFforBatch
from evaluations import mi_per_feature_discrete, computeEDF, prob_latent_given_group_hard, groupFairnessMeasures

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
X, y, S = load_census_data('data/adult.data',1)
# Define all the "intersectional groups"
intersectionalGroups = np.unique(S,axis=0) # all intersecting groups, i.e. black-women, white-man etc  
# load the test dataset
test_X, test_y, test_S = load_census_data('data/adult.test',0)

#X, y, S = sk.utils.shuffle(X, y, S, random_state=0)

#%% Simple example of a discrete latent variable (unsupervised NB) with multiple categorical observed variable
X=pd.concat([X, y], axis=1) 
test_X=pd.concat([test_X, test_y], axis=1) 

features = ['target', 'workclass_Local-gov', 'workclass_Never-worked',
       'workclass_Private', 'workclass_Self-emp-inc',
       'workclass_Self-emp-not-inc', 'workclass_State-gov',
       'workclass_Unknown', 'workclass_Without-pay', 'education_11th',
       'education_12th', 'education_1st-4th', 'education_5th-6th',
       'education_7th-8th', 'education_9th', 'education_Assoc-acdm',
       'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate',
       'education_HS-grad', 'education_Masters', 'education_Preschool',
       'education_Prof-school', 'education_Some-college','occupation_Armed-Forces', 'occupation_Craft-repair',
       'occupation_Exec-managerial', 'occupation_Farming-fishing',
       'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
       'occupation_Other-service', 'occupation_Priv-house-serv',
       'occupation_Prof-specialty', 'occupation_Protective-serv',
       'occupation_Sales', 'occupation_Tech-support',
       'occupation_Transport-moving', 'occupation_Unknown']

X = (X.loc[:, features])
test_X = (test_X.loc[:, features])

X = X.values 
S = S.values
test_X = test_X.values 
test_S = test_S.values

X, dev_X, S, dev_S = train_test_split(X, S, test_size=0.30,stratify=S, random_state=7)

X = torch.from_numpy(X)                     
testData = torch.from_numpy(test_X)
devData = torch.from_numpy(dev_X)

# concatenating an additional dimension before bernoulli income variable to represent it as categorical (low income and high income)
X = torch.cat((1-X[:,0].view(-1,1),X),1)
testData = torch.cat((1-testData[:,0].view(-1,1),testData),1)
devData = torch.cat((1-devData[:,0].view(-1,1),devData),1)

# held-out data conversion for evaluation
data_eval_train = X.numpy()
data_eval_test = testData.numpy()
data_eval_dev = devData.numpy()

testData = Variable(testData.float()).to(device)
devData = Variable(devData.float()).to(device)

#%% manually specifying the dimension of each categorical observed variables
income = [0,1]
workClass = list(range(2,9+1))
education = list(range(10,24+1)) 
occupation = list(range(25,38+1))
numObserv = [income, workClass, education, occupation] 

#%% deep neural network's hyperparameters
input_size = X.size()[1]
num_epochs = 10

latent_size = 1
cat_size = 2 # 2 latent class, i.e. hard-working and not hard-working

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

slack = 0.02 # small slack tolerance (1.0% degradation) in performance 
best_vanilla_ll = -44022.977 # log-likelihood of the best vanilla model on dev set
threshold_ll = best_vanilla_ll + best_vanilla_ll*slack # to pick allowable solutions
best_fairness = 0.031 # initialize with fairness of the best vanilla model for dev set

# for fairness interventions in vfae
PA_train = torch.from_numpy(S)                     
PA_test = torch.from_numpy(test_S)
PA_dev = torch.from_numpy(dev_S)

PA_test = Variable(PA_test.float()).to(device)
PA_dev = Variable(PA_dev.float()).to(device)

group_size = PA_train.size()[1]

#%% Report for grid-search experimental results on dev set
coulumn_names = ['ELBO', 'CLL','avg_mi', 'epsilon', 'gamma', 'avg_gamma','dp_race', 'dp_gender', 'dp_nation', 'pRule_race', 'pRule_gender', 'pRule_nation',\
                 'dp_race_avg', 'dp_gender_avg', 'dp_nation_avg', 'pRule_race_avg', 'pRule_gender_avg', 'pRule_nation_avg','overall_dp','overall_pRule']
Dev_results = np.zeros((numHypers,len(coulumn_names)))
Dev_results = pd.DataFrame(Dev_results, columns=coulumn_names)  

#%% Training discrete latent variable model
import sys
sys.stdout=open("results/gsVFAE_experimental_report.txt","w")

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
                            
                            bernoulli_vfae = BernoulliVFAE(input_size,latent_size,cat_size,group_size,encode_hidden,drop_prob,activation_func).to(device)
                            optimizer = optim.Adam(bernoulli_vfae.parameters(),lr = learning_rate,weight_decay=weight_decay)
                            
                            print(f"Trained with the following hyperparameters:")
                            print('RANDOM_STATE: ', RANDOM_STATE,'hidden layers (nodes/layer): ', encode_hidden, 'learning rate: ', learning_rate, 'Drop out prob: ', drop_prob, 'mini-batch sizes: ', minibatch, 'activation function: ', activation_func, 'weight decay: ', weight_decay)
                            print('\n')
                            
                            # training starts
                            batch_idx = 0
                            tau = 1.0 # initial temp
                            flag = False  # Defining the flag variable to avoid extreme solution arises from Gumbel softmax
                            bernoulli_vfae.train()
                            for epoch in range(num_epochs):
                                for batch in range(0,np.int64(np.floor(len(X)/minibatch))*minibatch,minibatch):
                                    trainS_batch = PA_train[batch:(batch+minibatch)]
                                    trainS_batch = Variable(trainS_batch.float()).to(device)
                                    
                                    trainX_batch = X[batch:(batch+minibatch)]
                                    trainX_batch = Variable(trainX_batch.float()).to(device)
                                    
                                    q_z, out_x = bernoulli_vfae(trainX_batch, trainS_batch, tau)
                                    
                                    elbo = vae_elbo_function(trainX_batch, out_x, numObserv, q_z, latent_prior)
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
                                    qz_typ, typ_out_x = bernoulli_vfae(devData, PA_dev, tau)
                                Dev_results['ELBO'][run]=vae_elbo_function(devData, typ_out_x, numObserv, qz_typ, latent_prior)
                                Dev_results['CLL'][run] = vae_held_out_logLL(devData, typ_out_x, numObserv, qz_typ)

                                if qz_typ.device.type == 'cuda':
                                    qz_typ = qz_typ.cpu()
                                qz_typ = np.array(qz_typ)
                                hard_cluster = np.argmax(qz_typ, axis=1)
                                
                                mi_per_feature = mi_per_feature_discrete(data_eval_dev, hard_cluster, numObserv)
                                Dev_results['avg_mi'][run] = np.mean(mi_per_feature)
                                Dev_results['epsilon'][run], Dev_results['gamma'][run], Dev_results['avg_gamma'][run] = computeEDF(dev_S,qz_typ,intersectionalGroups)
                                Dev_results['dp_race'][run], Dev_results['dp_gender'][run], Dev_results['dp_nation'][run], Dev_results['pRule_race'][run], Dev_results['pRule_gender'][run], Dev_results['pRule_nation'][run], Dev_results['dp_race_avg'][run], Dev_results['dp_gender_avg'][run], Dev_results['dp_nation_avg'][run], Dev_results['pRule_race_avg'][run], Dev_results['pRule_gender_avg'][run], Dev_results['pRule_nation_avg'][run] = groupFairnessMeasures(dev_S,qz_typ)
                                Dev_results['overall_dp'][run] = (Dev_results['dp_race_avg'][run]+Dev_results['dp_gender_avg'][run]+Dev_results['dp_nation_avg'][run])/dev_S.shape[1]
                                Dev_results['overall_pRule'][run] = (Dev_results['pRule_race_avg'][run]+Dev_results['pRule_gender_avg'][run]+Dev_results['pRule_nation_avg'][run])/dev_S.shape[1]

                                # choose the best model based on a pre-defined condition
                                if Dev_results['CLL'][run]>=threshold_ll and Dev_results['overall_dp'][run]<best_fairness and len(np.unique(hard_cluster))>1:
                                    best_fairness = Dev_results['overall_dp'][run]
                                    print('\n')
                                    print(f"Current best model:{run: d}")
                                    print('RANDOM_STATE: ', RANDOM_STATE,'hidden layers (nodes/layer): ', encode_hidden, 'learning rate: ', learning_rate, 'Drop out prob: ', drop_prob, 'mini-batch sizes: ', minibatch, 'activation function: ', activation_func, 'weight decay: ', weight_decay)
                                    print(f"LL on dev set: {Dev_results['CLL'][run]: .3f}")
                                    print(f"Overall DP on dev set: {Dev_results['overall_dp'][run]: .3f}")
                                    best_net = bernoulli_vfae
                                    print('\n')
                                    print('\n')
                            
                            run += 1


#%% Final evaluation on the best model
print('\n')
print('For the selected best model,')

Dev_results.to_csv('results/Dev_results_gsVFAE.csv',index=False)

print('Evaluation results on the test set:')
print('\n')

with torch.no_grad():
    qz_typ, typ_out_x = best_net(testData, PA_test, tau)

test_elbo_typ=vae_elbo_function(testData, typ_out_x, numObserv, qz_typ, latent_prior)
print(f"average ELBO: {test_elbo_typ: .3f}")
test_ll_typ = vae_held_out_logLL(testData, typ_out_x, numObserv, qz_typ)
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
dp_race, dp_gender, dp_nation, pRule_race, pRule_gender, pRule_nation, dp_race_avg, dp_gender_avg, dp_nation_avg, pRule_race_avg, pRule_gender_avg, pRule_nation_avg = groupFairnessMeasures(test_S,qz_typ)

print(f"Demographic parity measures (race): {dp_race: .3f}")
print(f"Demographic parity measures (gender): {dp_gender: .3f}")
print(f"Demographic parity measures (nationality): {dp_nation: .3f}")

print(f"p%-rule measures (race): {pRule_race: .3f}")
print(f"p%-rule measures (gender): {pRule_gender: .3f}")
print(f"p%-rule measures (nationality): {pRule_nation: .3f}")

print(f"Avg Demographic parity measures (race): {dp_race_avg: .3f}")
print(f"Avg Demographic parity measures (gender): {dp_gender_avg: .3f}")
print(f"Avg Demographic parity measures (nationality): {dp_nation_avg: .3f}")

print(f"Avg p%-rule measures (race): {pRule_race_avg: .3f}")
print(f"Avg p%-rule measures (gender): {pRule_gender_avg: .3f}")
print(f"Avg p%-rule measures (nationality): {pRule_nation_avg: .3f}")

overall_dp = (dp_race_avg+dp_gender_avg+dp_nation_avg)/test_S.shape[1]
overall_pRule = (pRule_race_avg+pRule_gender_avg+pRule_nation_avg)/test_S.shape[1]

print(f"Overall Demographic parity measures: {overall_dp: .3f}")
print(f"Overall p%-rule measures: {overall_pRule: .3f}")
# mutual info
mi_per_feature = mi_per_feature_discrete(data_eval_test, hard_cluster, numObserv)
print(f"avg mi: {np.mean(mi_per_feature): .3f}")
np.savetxt('results/vfae_mi_per_feature_gsVFAE.txt',mi_per_feature)

#%% saving model and results
torch.save(best_net.state_dict(), "trained-models/gs_vfae")