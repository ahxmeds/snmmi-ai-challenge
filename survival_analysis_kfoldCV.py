#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.ensemble import RandomSurvivalForest
import seaborn as sns
from sksurv.util import Surv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.feature_selection import SequentialFeatureSelector
import time 

#%%
def create_sksurv_labels(data_y_):
    survival_times = data_y_['Outcome_PFS'].values
    event_indicators = data_y_['Event'].values
    y = np.empty(len(survival_times), dtype=[('Status', bool), ('Survival_in_months', float)])
    y['Status'] = event_indicators
    y['Survival_in_months'] = survival_times
    return y

def feature_selection_spearmanr_correlation(X):
    correlation_matrix = np.zeros((X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            correlation, _ = spearmanr(X[:, i], X[:, j])
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
    threshold = 0.6
    high_corr_indices = np.where(np.abs(correlation_matrix) > threshold) 

    high_corr_features = set()
    for i, j in zip(*high_corr_indices):
        if i != j and i < j:
            high_corr_features.add(j)

    X_reduced = np.delete(X, list(high_corr_features), axis=1)
    return X_reduced

#%%
train_data_fname = '/home/jhubadmin/Projects/snmmi-ai-challenge/SNMMI_CHALLENGE_TRAINING_V22OCT2023.xlsx'
train_data = pd.read_excel(train_data_fname)
train_x = train_data.iloc[:, 3:]
train_y = train_data.iloc[:, 1:3]
y_train = create_sksurv_labels(train_y)
print('Data loaded')
#%%
cox = CoxPHSurvivalAnalysis(alpha=2.33)
num_features = 20
selector = SequentialFeatureSelector(cox, n_features_to_select=num_features, direction='forward')
pipe = make_pipeline(
    SimpleImputer(strategy='mean'),  
    StandardScaler(),
)
#%%
# impute and scale
print('feature scaling started')
t1 = time.time()
X_train_scaled = pipe.fit_transform(train_x)
print(f'features scaled ended: {time.time() - t1} seconds')
#%%
# do feature selection
print('feature selection started')
t2 = time.time()
X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
print(f'features selection ended: {(time.time() - t2)/60} min')
#%%
np.save(f'MeanImpute_StandardScaler_SequentialFeatureSelectionForwardn{num_features}.npy', X_train_reduced)

#%%
t3 = time.time()

n_splits = [5]
mean_c_indices = []
for n in n_splits:
    k_fold = KFold(n_splits=n)
    c_index_list = []
    for train_index, valid_index in k_fold.split(X_train_reduced):
        X_train_, X_valid_ = X_train_reduced[train_index], X_train_reduced[valid_index]
        y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
        
        cox.fit(X_train_, y_train_)
        
        # Calculate concordance index for evaluation
        prediction = cox.predict(X_valid_)
        c_index = concordance_index_censored(y_valid_['Status'], y_valid_['Survival_in_months'], prediction)
        c_index_list.append(c_index[0])

    # print(f'{n} fold CV training ended {(time.time() - t3)/60} mins')
    mean_c_index = sum(c_index_list)/n
    print(f"Nsplits: {n}; Mean Concordance Index: {mean_c_index}")
    mean_c_indices.append(mean_c_index)


# %%
plt.plot(n_splits, mean_c_indices)
# %%
for n in n_splits:
    k_fold = KFold(n_splits=n)
    c_index_list = []
    for train_index, valid_index in k_fold.split(X_train_reduced):
        X_train_, X_valid_ = X_train_reduced[train_index], X_train_reduced[valid_index]
        y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
        
        cox.fit(X_train_, y_train_)
        
        # Calculate concordance index for evaluation
        prediction = cox.predict(X_valid_)
        c_index = concordance_index_censored(y_valid_['Status'], y_valid_['Survival_in_months'], prediction)
        c_index_list.append(c_index[0])

    # print(f'{n} fold CV training ended {(time.time() - t3)/60} mins')
    mean_c_index = sum(c_index_list)/n
    print(f"Nsplits: {n}; Mean Concordance Index: {mean_c_index}")
    mean_c_indices.append(mean_c_index)
