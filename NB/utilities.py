
import pandas as pd
import numpy as np

def load_census_data (path,check):
    column_names = ['age', 'workclass','fnlwgt','education','education_num',
                    'marital_status','occupation','relationship','race','gender',
                    'capital_gain','capital_loss','hours_per_week','nationality','target']
    input_data = (pd.read_csv(path,names=column_names,
                               na_values="?",sep=r'\s*,\s*',engine='python'))
    # sensitive attributes; we identify 'race','gender' and 'nationality' as sensitive attributes
    # note : keeping the protected attributes in the data set, but make sure they are converted to same category as in the S
    input_data['race'] = input_data['race'].map({'Black': 0,'White': 1, 'Asian-Pac-Islander': 0, 'Amer-Indian-Eskimo': 0, 'Other': 0})
    input_data['gender'] = (input_data['gender'] == 'Male').astype(int)
    input_data['nationality'] = (input_data['nationality'] == 'United-States').astype(int)
    
    protected_attribs = ['race', 'gender','nationality'] # remove nationality from protected attribute
    S = (input_data.loc[:, protected_attribs])
    
# =============================================================================
#     # merge few categories to reduce curse of dimensionality
#     input_data['workclass'] = input_data['workclass'].map({'Local-gov': 1,'Never-worked': 0, 'Private': 2, 'Self-emp-inc': 2, 'Self-emp-not-inc': 2, 'State-gov': 1,'Unknown': 0, 'Without-pay': 0})
#     input_data['education'] = input_data['education'].map({'11th': 3,'12th': 3, '1st-4th': 0, '5th-6th': 0, '7th-8th': 0, '9th': 3,'Assoc-acdm': 4, 'Assoc-voc': 4, 'Bachelors': 2, 'Doctorate': 1, 'HS-grad': 1,'Masters': 1, 'Preschool': 0, 'Prof-school': 4, 'Some-college': 3}) 
#     input_data['occupation'] = input_data['occupation'].map({'Armed-Forces': 2,'Craft-repair': 0, 'Exec-managerial': 1, 'Farming-fishing': 2, 'Handlers-cleaners': 0, 'Machine-op-inspct': 0,'Other-service': 0, 'Priv-house-serv': 2, 'Prof-specialty': 1, 'Protective-serv': 2, 'Sales': 2,'Tech-support': 1, 'Transport-moving': 0, 'Unknown': 0})
# =============================================================================
    # targets; 1 when someone makes over 50k , otherwise 0
    if(check):
        y = (input_data['target'] == '>50K').astype(int)    # target 1 when income>50K
    else:
        y = (input_data['target'] == '>50K.').astype(int)    # target 1 when income>50K
    
    X = (input_data
         .drop(columns=['target','race', 'gender','nationality','relationship','marital_status'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    return X, y, S
