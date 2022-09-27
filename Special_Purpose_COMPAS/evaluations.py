import numpy as np


#%% P(z|A)
def prob_latent_given_group(protectedAttributes,intersectGroups,qz):
    numClasses = qz.shape[1]
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    qz_per_group = np.zeros((len(intersectGroups),qz.shape[1]))
    population_per_group = np.zeros((len(intersectGroups),qz.shape[1]))
    for i in range(len(qz)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        qz_per_group[index,:] = qz_per_group[index,:] + qz[i,:]  
        population_per_group[index,:] = population_per_group[index,:] + 1        
    return (qz_per_group + dirichletAlpha)/(population_per_group + concentrationParameter)
#%% Measure \epsilon-DF from positive predict probabilities
def differentialFairnessForCatVar(probabilitiesOfPositive,eps=1e-20):
    # input: probabilitiesOfPositive = positive p(z|S) from inference network
    # output: epsilon = differential fairness measure
    epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = 0.0 # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                for y in range(probabilitiesOfPositive.shape[1]):
                    epsilon = max(epsilon,abs(np.log(probabilitiesOfPositive[i,y])-np.log(probabilitiesOfPositive[j,y])))# ratio of probabilities of positive outcome
                    #epsilon = max(epsilon,abs(np.log((1-probabilitiesOfPositive[i,y]))-np.log((1-probabilitiesOfPositive[j,y])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

#%%
# Measure \gamma-SF (gamma unfairness) 
def subgroupFairnessForCatVar(probabilitiesOfPositive_all,alphaSP_all):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    gamma = 0.0 # initialization of SF
    gamma_avg = 0.0
    for y in range(probabilitiesOfPositive_all.shape[1]):
        probabilitiesOfPositive = probabilitiesOfPositive_all[:,y]
        alphaSP = alphaSP_all[:,y]
        spD = sum(probabilitiesOfPositive*alphaSP) # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
        gammaPerGroup = np.zeros(len(probabilitiesOfPositive)) # SF per group
        for i in range(len(probabilitiesOfPositive)):
            gammaPerGroup[i] = alphaSP[i]*abs(spD-probabilitiesOfPositive[i])
        gamma_avg += max(gammaPerGroup)
        gamma = max(gamma,max(gammaPerGroup)) # overall SF of the algorithm 
    return gamma, gamma_avg/probabilitiesOfPositive_all.shape[1]

#%% intersectional fairness measurement from smoothed empirical counts 
def computeEDF(protectedAttributes,latent_distribution,intersectGroups):
    # compute counts and probabilities
    countsClassOne = np.zeros((len(intersectGroups),latent_distribution.shape[1]))
    countsTotal = np.zeros((len(intersectGroups),latent_distribution.shape[1]))
    
    numClasses = latent_distribution.shape[1]
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    for i in range(len(latent_distribution)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index,:] = countsTotal[index,:] + 1.0
        countsClassOne[index,:] = countsClassOne[index,:] + latent_distribution[i,:]        
    
    # probability of z given S (p(z=1|S)): probability distribution over merit per value of the protected attributes
    probabilitiesOfPositive = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter)
    alphaG_smoothed = (countsTotal + dirichletAlpha) /(len(latent_distribution) + concentrationParameter)

    epsilon = differentialFairnessForCatVar(probabilitiesOfPositive)
    gamma, gamma_avg = subgroupFairnessForCatVar(probabilitiesOfPositive, alphaG_smoothed)
    
    return epsilon, gamma, gamma_avg

#%% group fairness measurement from smoothed empirical counts for Adult dataset 
def demographicParity(binaryGroup, per_class_z, dirichletAlpha, concentrationParameter):

    """ It is often impossible or undesirable to satisfy demographic parity exactly 
        (i.e. achieve complete independence).
        In this case, a useful metric is demographic parity distance """

    non_prot_all = sum(binaryGroup == 1) # privileged group
    prot_all = sum(binaryGroup == 0) # unprivileged group
    
    non_prot_pos_soft = sum(per_class_z[binaryGroup == 1]) # privileged in positive class
    prot_pos_soft = sum(per_class_z[binaryGroup == 0]) # unprivileged in positive class
    frac_non_prot_pos_soft = float(non_prot_pos_soft+dirichletAlpha) / float(non_prot_all+concentrationParameter)
    frac_prot_pos_soft = float(prot_pos_soft+dirichletAlpha) / float(prot_all+concentrationParameter)
    
    # demographic parity distance
    dp_soft = max(abs(frac_prot_pos_soft-frac_non_prot_pos_soft), abs(frac_non_prot_pos_soft-frac_prot_pos_soft))
    
    # p% - rule
    p_rule_soft = min(frac_prot_pos_soft / frac_non_prot_pos_soft,frac_non_prot_pos_soft / frac_prot_pos_soft) * 100.0
    
    return dp_soft,p_rule_soft

def groupFairnessMeasures(protectedAttributes,latent_distribution):
    S_race = np.int64(np.zeros((len(protectedAttributes))))
    S_gender = np.int64(np.zeros((len(protectedAttributes))))
    for i in range(len(protectedAttributes)):
        if protectedAttributes[i,0]==1:
            S_race[i] = 1
        if protectedAttributes[i,1]==1:
            S_gender[i] = 1
    # smoothing parameter
    numClasses = latent_distribution.shape[1]
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    # group fairness for race
    dp_race = 0.0
    pRule_race = 100.0

    dp_race_avg = 0.0
    pRule_race_avg = 0.0
    for y in range(latent_distribution.shape[1]):
        dp_temp, pRule_temp = demographicParity(S_race, latent_distribution[:,y], dirichletAlpha, concentrationParameter)
        dp_race_avg += dp_temp
        pRule_race_avg += pRule_temp
        dp_race = max(dp_race,dp_temp)
        pRule_race = min(pRule_race,pRule_temp)

    # group fairness for gender
    dp_gender = 0.0
    pRule_gender = 100.0
    
    dp_gender_avg = 0.0
    pRule_gender_avg = 0.0
    for y in range(latent_distribution.shape[1]):
        dp_temp, pRule_temp = demographicParity(S_gender, latent_distribution[:,y], dirichletAlpha, concentrationParameter)
        dp_gender_avg += dp_temp
        pRule_gender_avg += pRule_temp
        dp_gender = max(dp_gender,dp_temp)
        pRule_gender = min(pRule_gender,pRule_temp)    

    return dp_race,dp_gender,pRule_race,pRule_gender,dp_race_avg/numClasses,dp_gender_avg/numClasses,pRule_race_avg/numClasses,pRule_gender_avg/numClasses

#%% without EDF
def prob_latent_given_group_hard(protectedAttributes,intersectGroups,qz):
    qz_per_group = np.zeros((len(intersectGroups),qz.shape[1]))
    population_per_group = np.zeros((len(intersectGroups),qz.shape[1]))
    for i in range(len(qz)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        qz_per_group[index,:] = qz_per_group[index,:] + qz[i,:]  
        population_per_group[index,:] = population_per_group[index,:] + 1        
    return qz_per_group /population_per_group 


def computeDF_hard(protectedAttributes,latent_distribution,intersectGroups):
    # compute counts and probabilities
    countsClassOne = np.zeros((len(intersectGroups),latent_distribution.shape[1]))
    countsTotal = np.zeros((len(intersectGroups),latent_distribution.shape[1]))
    
    
    for i in range(len(latent_distribution)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index,:] = countsTotal[index,:] + 1.0
        countsClassOne[index,:] = countsClassOne[index,:] + latent_distribution[i,:]        
    
    # probability of z given S (p(z=1|S)): probability distribution over merit per value of the protected attributes
    probabilitiesOfPositive = countsClassOne /countsTotal 

    epsilon = differentialFairnessForCatVar(probabilitiesOfPositive)
    
    return epsilon