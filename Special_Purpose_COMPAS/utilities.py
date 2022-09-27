import pandas as pd
import numpy as np

def load_compas_data (path):
    input_data = (pd.read_csv(path))
    
    features = ['sex','race','age_cat','c_charge_degree','score_text','priors_count','days_b_screening_arrest','c_days_from_compas','decile_score','is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out','juv_fel_count','juv_misd_count','juv_other_count']
                
    x_data = (input_data.loc[:, features])
    X=x_data.dropna().reset_index(drop=True)
    
    # pre-process juvenile felony and misdemeanor counts
    juv_fel = (X.juv_fel_count).rename('juv_fel')
    juv_misd = (X.juv_misd_count+X.juv_other_count).rename('juv_misd')

    # for evaluating latent variable models, we will use the categorical compas score and actual recidivism (two_year_recid)
    X['score_text'] = X['score_text'].map({'Low': 0,'Medium': 1,'High':2})
    compas_pred = X.score_text
    actual_recid = X.two_year_recid

    # concatenate the variables correspond to latent class z
    #,X.score_text,
    
    # priors_count: counts of prior convictions
    # days_b_screening_arrest: Days between Screening and Arrest
    # juvenile criminal activity: NUMBER OF JUVENILE CHARGES (FELONY, MISDEMEANOR, OTHER)
    X_z = pd.concat([X.decile_score,X.priors_count,juv_fel,juv_misd],axis=1)
    
    # jail time is the observed variable which depends on observed charge degree, and latent class z and u
    jail_time=((pd.to_datetime(X.c_jail_out)-pd.to_datetime(X.c_jail_in)).dt.total_seconds()/86400).rename('jail_time')
    jail_time[jail_time<0] = 0 # when length of stay in jail is negative, we map it to zero
    
    # concatenate the variables correspond to only the latent class u
    X['c_charge_degree'] = X['c_charge_degree'].map({'M': 'misd','F': 'violent'})
    X['age_cat'] = X['age_cat'].map({'Less than 25': 'a_young','25 - 45': 'midlevel', 'Greater than 45':'old'})
# =============================================================================
#     X_u_temp = pd.concat([X.c_charge_degree,X.age_cat], axis =1)
#     X_u = pd.get_dummies(X_u_temp)
# =============================================================================
    X_age = pd.get_dummies(X.age_cat)
    X_charge = pd.get_dummies(X.c_charge_degree)
    
    # pre-process protected attributes
    protectedFeatures = ['race', 'sex']
    X['race'] = X['race'].map({'African-American': 0,'Caucasian': 1, 'Hispanic': 0, 'Native American': 0,'Asian': 0, 'Other': 0})
    X['sex'] = X['sex'].map({'Female': 0,'Male': 1})
    S = (X.loc[:, protectedFeatures])
    
    return X_z, X_age, X_charge, jail_time, S, compas_pred, actual_recid
