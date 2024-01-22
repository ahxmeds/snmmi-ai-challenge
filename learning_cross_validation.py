#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
# %%
def create_sksurv_labels(data_y_):
    survival_times = data_y_['Outcome_PFS'].values
    event_indicators = data_y_['Event'].values
    y = np.empty(len(survival_times), dtype=[('Status', bool), ('Survival_in_months', float)])
    y['Status'] = event_indicators
    y['Survival_in_months'] = survival_times
    return y

train_data  = pd.read_excel('/home/skurkowska/SNMMI_challenge_ML/SNMMI_CHALLENGE_TRAINING_V22OCT2023.xlsx')
train_x = train_data.iloc[:, 3:]
train_y_ = train_data.iloc[:, 1:3]

# %%
train_y = create_sksurv_labels(train_y_)

#%%
# IN CROSS-VALIDATION WE DO NOT SPLIT TO HAVE TRAIN AND VALID BECAUSE WE WILL HAVE MULTIPLE OF THOSE
# X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# %%
pipe = make_pipeline(
    SimpleImputer(strategy='mean'),  
    StandardScaler()
)
X_scaled = pipe.fit_transform(train_x)
#X_valid_scaled = pipe.transform(train_x)


# %%
from scipy.stats import spearmanr
# Compute Spearman correlation matrix for training data
correlation_matrix = np.zeros((X_scaled.shape[1], X_scaled.shape[1]))

for i in range(X_scaled.shape[1]):
    for j in range(i + 1, X_scaled.shape[1]):
        correlation, _ = spearmanr(X_scaled[:, i], X_scaled[:, j])
        correlation_matrix[i, j] = correlation
        correlation_matrix[j, i] = correlation

# Find pairs of highly correlated features
high_corr_indices = np.where(np.abs(correlation_matrix) > 0.8)  # You can adjust the threshold as needed

# Create a set of columns to drop
high_corr_features = set()
for i, j in zip(*high_corr_indices):
    if i != j and i < j:
        high_corr_features.add(j)

# Drop redundant features from training data
X_reduced = np.delete(X_scaled, list(high_corr_features), axis=1)

# Drop the same features from the test dataset as from training data
#X_valid_reduced = np.delete(X_valid_scaled, list(high_corr_features), axis=1)

#%%
from sklearn.model_selection import KFold, cross_val_score


##########################################################
################# 1. DEFINE THE MODEL ###################
##########################################################
alpha = 2.33
cph = CoxPHSurvivalAnalysis(alpha = alpha)


##########################################################
############### 2. DEFINE THE CV METHOD ##################
##########################################################
kf = KFold(n_splits=5, shuffle=True, random_state=42)


##########################################################
#################### 3. PERFORM CV #######################
##########################################################

c_index_scores = cross_val_score(cph, X_reduced, train_y, cv=kf)

# Print the average C-index across folds
print("Average C-index:", np.mean(c_index_scores))

#### Average C-index: 0.6479424945629498
#### Result withou5 CV: 0.6683725690890481

