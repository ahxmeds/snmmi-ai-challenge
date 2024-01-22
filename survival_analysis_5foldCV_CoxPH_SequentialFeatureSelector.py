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
#%%
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
selected_feature_indices = selector.get_support(indices=True)
selected_features = train_x.columns[selected_feature_indices]
print("Selected Features:", selected_features)

#%%
np.save(f'CoxPH_MeanImpute_StandardScaler_SequentialFeatureSelectionForwardn{num_features}.npy', X_train_reduced)
X_train_reduced = np.load(f'CoxPH_MeanImpute_StandardScaler_SequentialFeatureSelectionForwardn{num_features}.npy')

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


#%%
# fit on the entire training set
cox = CoxPHSurvivalAnalysis(alpha=2.33)
cox.fit(X_train_reduced, y_train)
prediction = cox.predict(X_train_reduced)
c_index = concordance_index_censored(y_train['Status'], y_train['Survival_in_months'], prediction)
# %%

test_data_fname = '/home/jhubadmin/Projects/snmmi-ai-challenge/SNMMI_CHALLENGE_TESTING_V01112023.xlsx'
test_data = pd.read_excel(test_data_fname)
test_x = test_data.iloc[:, 1:]
X_test_scaled = pipe.transform(test_x)
X_test_reduced = selector.transform(X_test_scaled)

#%%
test_reporting_classification_fname = '/home/jhubadmin/Projects/snmmi-ai-challenge/SNMMI_CHALLENGE_TESTING_REPORTFORM_ClassificationOutcome.xlsx'
clf_report = pd.read_excel(test_reporting_classification_fname)

#%%%
# train_y = train_data.iloc[:, 1:3]
survival_functions = cox.predict_survival_function(X_test_reduced)

# Define time points for which you want to extract survival probabilities
time_points = [12, 24, 36]  # Time points in months (1 year, 2 years, 3 years)
Probs_year1, Probs_year2, Probs_year3 = [], [], []
# Iterate over patients and time points to get survival probabilities
for patient_index, survival_function in enumerate(survival_functions):
    patient_id = test_data.iloc[patient_index, 0]  # Assuming the first column contains patient IDs
    print(f"Patient ID: {patient_id}")

    for time_point in time_points:
        # Extract survival probability at the specified time point
        survival_probability = survival_function(time_point)
        if time_point == 12:
            Probs_year1.append(survival_probability)
        elif time_point == 24:
            Probs_year2.append(survival_probability)
        elif time_point == 36:
            Probs_year3.append(survival_probability)
        else:
            pass
        print(f"Survival Probability at {time_point} months: {survival_probability:.4f}")

    print("\n")
# %%
column_names = [
    'PatientID',
    'Probability for 1 year PFS (0-1.0)',
    'Probability for 2 year PFS (0-1.0)',
    'Probability for 3 year PFS (0-1.0)'
]
patient_ids = clf_report['PatientID'].tolist()

test_clf_data = np.column_stack((patient_ids, Probs_year1, Probs_year2, Probs_year3))
clf_df = pd.DataFrame(test_clf_data, columns=column_names)
clf_df.to_excel('SNMMI_CHALLENGE_TESTING_REPORTFORM_ClassificationOutcome.xlsx', index=False)
# %%
def find_pfs_at_survival_probability_threshold_T(survival_function, T=0.5):
    array_x = survival_function.x
    array_y = survival_function.y

    index_at_T = np.argmin(np.abs(array_y - T))
    PFS_at_T = array_x[index_at_T]
    return PFS_at_T, index_at_T
 
def get_pfs(y):
    pfs_list = []
    for i in range(len(y)):
        pfs_list.append(y[i][1])
    return pfs_list 
#%%
def mean_squared_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ratio = (y_true - y_pred)/y_true
    mspe = np.mean(ratio**2)
    return mspe

def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = (y_true - y_pred)
    mse = np.mean(diff**2)
    return mse


n_splits = [5]
T_list = np.linspace(0.001, 0.999, 30)
MEAN_MSPE = []
MEAN_MSE = []
for T in T_list:
    mspe_list = []
    mse_list = []
    for n in n_splits:
        k_fold = KFold(n_splits=n)
        for train_index, valid_index in k_fold.split(X_train_reduced):
            X_train_, X_valid_ = X_train_reduced[train_index], X_train_reduced[valid_index]
            y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
            
            cox.fit(X_train_, y_train_)
            
            # Calculate concordance index for evaluation
            survival_functions = cox.predict_survival_function(X_valid_)
            y_pfs = get_pfs(y_valid_)
            pred_pfs = []
            for index, surv_func in enumerate(survival_functions):
                pfs, _ = find_pfs_at_survival_probability_threshold_T(surv_func, T)
                pred_pfs.append(pfs)
            mspe = mean_squared_percentage_error(y_pfs, pred_pfs)
            mse = mean_squared_error(y_pfs, pred_pfs)
            mspe_list.append(mspe)
            mse_list.append(mse)
    MEAN_MSPE.append(np.mean(mspe_list))
    MEAN_MSE.append(np.mean(mse_list))

fig, ax = plt.subplots(1,2)
ax[0].plot(T_list, MEAN_MSE, label='MSE')
ax[1].plot(T_list, MEAN_MSPE, label='MSPE')
ax[0].legend()
ax[1].legend()
plt.show()
#%%

# %%
#%%
# fit on the entire training set
cox = CoxPHSurvivalAnalysis(alpha=2.33)
cox.fit(X_train_reduced, y_train)
survival_functions = cox.predict_survival_function(X_train_reduced)
y_pfs = get_pfs(y_train)
pred_pfs = []
for index, surv_func in enumerate(survival_functions):
    pfs, _ = find_pfs_at_survival_probability_threshold_T(surv_func, best_T)
    pred_pfs.append(pfs)
mspe = mean_squared_percentage_error(y_pfs, pred_pfs)
mse =  mean_squared_error(y_pfs, pred_pfs)
print(mspe)
print(mse)

# %%
best_T = T_list[np.argmin(MEAN_MSE)]
test_reporting_regression_fname = '/home/jhubadmin/Projects/snmmi-ai-challenge/SNMMI_CHALLENGE_TESTING_REPORTFORM_ContinuousOutcomes.xlsx'
reg_report = pd.read_excel(test_reporting_regression_fname)
survival_functions = cox.predict_survival_function(X_test_reduced)
predicted_pfs = []
for index, surv_func in enumerate(survival_functions):
    pfs, _ = find_pfs_at_survival_probability_threshold_T(surv_func, best_T)
    predicted_pfs.append(pfs)

column_names = [
    'PatientID',
    'Predicted PFS (months, decimals are allowed)'
]

patient_ids = reg_report['PatientID'].tolist()

test_reg_data = np.column_stack((patient_ids, predicted_pfs))
reg_df = pd.DataFrame(test_reg_data, columns=column_names)
reg_df.to_excel('SNMMI_CHALLENGE_TESTING_REPORTFORM_ContinuousOutcome.xlsx', index=False)
# %%
