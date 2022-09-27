import pandas as pd
import numpy as np

#%% Health heritage prize (HHP) data
def load_hhp_data(path):
    data = pd.read_csv(path)
    sex = data['sexMISS'] == 0
    age = data['age_MISS'] == 0
    
    data = data[sex & age]
    data = data.drop(['DaysInHospital','MemberID_t','trainset','sexMISS','age_MISS'], axis=1)
    # Year 3 of 'DaysInHospital' is not complete (all intances are nan), so we drop this column
    data['YEAR_t'] = data['YEAR_t'].map({'Y1': 0,'Y2': 1, 'Y3': 2})
    
    ages = (data[['age_%d5' % (i) for i in range(0, 9)]]).values
    sexs = (data[['sexMALE', 'sexFEMALE']]).values
    # ages 85 encoded by 0 and sex female encoded by 0
    
    uniqueAges = np.unique(ages,axis=0)
    uniqueSexs = np.unique(sexs,axis=0)
    
    protected_attr = np.int64(np.zeros((len(data),2)))
    
    # for protected attributes in categorical
    for i in range(len(data)):
        indx_sex=np.where((uniqueSexs==sexs[i]).all(axis=1))[0][0]
        protected_attr[i,0] = indx_sex
        
        indx_age=np.where((uniqueAges==ages[i]).all(axis=1))[0][0]
        protected_attr[i,1] = indx_age
    
    # binary age: age>= 65 is unprivileged
    protected_attr[:,1] = np.int64(protected_attr[:,1]>2)
        
    # drop all demographic info related attributes
    data = data.drop(['age_05','age_15','age_25','age_35','age_45','age_55','age_65','age_75','age_85','sexMALE', 'sexFEMALE'], axis=1)
    return data, protected_attr
