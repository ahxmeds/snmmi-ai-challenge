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

# %%
def create_sksurv_labels(data_y_):
    survival_times = data_y_['Outcome_PFS'].values
    event_indicators = data_y_['Event'].values
    y = np.empty(len(survival_times), dtype=[('Status', bool), ('Survival_in_months', float)])
    y['Status'] = event_indicators
    y['Survival_in_months'] = survival_times
    return y

train_data_fname = '/home/jhubadmin/Projects/snmmi-ai-challenge/SNMMI_CHALLENGE_TRAINING_V22OCT2023.xlsx'
#%%
train_data = pd.read_excel(train_data_fname)
train_x = train_data.iloc[:, 3:]
train_y_ = train_data.iloc[:, 1:3]

#%%
correlation_matrix = train_x.corr()

# Generate a heatmap without annotations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Features')
plt.show()
#%%
train_y = create_sksurv_labels(train_y_)
#%%
X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

#%%
pipe = make_pipeline(
    SimpleImputer(strategy='mean'),  
    StandardScaler()
)
X_train_scaled = pipe.fit_transform(X_train)
X_valid_scaled = pipe.transform(X_valid)

#%%
import pandas as pd
from scipy.stats import spearmanr

# Compute Spearman correlation matrix for training data
correlation_matrix = np.zeros((X_train_scaled.shape[1], X_train_scaled.shape[1]))

for i in range(X_train_scaled.shape[1]):
    for j in range(i + 1, X_train_scaled.shape[1]):
        correlation, _ = spearmanr(X_train_scaled[:, i], X_train_scaled[:, j])
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
X_train_reduced = np.delete(X_train_scaled, list(high_corr_features), axis=1)

# Drop the same features from the test dataset as from training data
X_valid_reduced = np.delete(X_valid_scaled, list(high_corr_features), axis=1)
#%%
##
# Cox Proportionality Hazard (alpha is hyperparameter)
alpha = 2.33
cph = CoxPHSurvivalAnalysis(alpha = alpha)
cph.fit(X_train_reduced, y_train)
survival_functions = cph.predict_survival_function(X_valid_reduced)
y_valid_surv = Surv.from_arrays(y_valid['Status'], y_valid['Survival_in_months'])

#%%
predicted_pfs_durations = []
threshold = 0.5  # Adjust threshold if needed

for sf in survival_functions:
    # Find time when survival function drops below threshold
    pfs_time = next((t for t, s in zip(sf.x, sf.y) if s <= threshold), sf.x[-1])  # Set maximum duration if no drop below threshold
    predicted_pfs_durations.append(pfs_time)

actual_pfs_durations = [event[1] for event in y_valid_surv]
mse = mean_squared_error(actual_pfs_durations, predicted_pfs_durations)

#%%
prediction = cph.predict(X_valid_reduced)
c_index_censored = concordance_index_censored(y_valid["Status"], y_valid["Survival_in_months"], prediction)[0]
print(c_index_censored)

