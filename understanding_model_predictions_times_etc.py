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
from sklearn.feature_selection import SequentialFeatureSelector

# %%
# %%
def create_sksurv_labels(data_y_):
    survival_times = data_y_['Outcome_PFS'].values
    event_indicators = data_y_['Event'].values
    y = np.empty(len(survival_times), dtype=[('Status', bool), ('Survival_in_months', float)])
    y['Status'] = event_indicators
    y['Survival_in_months'] = survival_times
    return y

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
X_train_reduced = np.load('MeanImpute_StandardScaler_SequentialFeatureSelectionCoxForwardn20.npy')

# %%

cox.fit(X_train_reduced, y_train)
# %%
predictions = cox.predict(X_train_reduced)
baseline_survival_function = cox.baseline_survival_
# %%
survival_function = cox.predict_survival_function(X_train_reduced)
# %%
survival_function_pt1 = survival_function[0]
plt.step(baseline_survival_function.x, baseline_survival_function.y)
plt.step(survival_function_pt1.x, survival_function_pt1.y)
# %%
exp_predictions = np.exp(predictions)
calculated_survival_function_pt1 = baseline_survival_function.y*exp_predictions[0]

# %%
plt.step(baseline_survival_function.x, baseline_survival_function.y, label='Baseline')
plt.step(survival_function_pt1.x, survival_function_pt1.y, label='True')
plt.step(survival_function_pt1.x, calculated_survival_function_pt1, label='Manual')
plt.legend()
# %%
