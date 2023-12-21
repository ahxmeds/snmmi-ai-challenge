#%%
import pandas as pd 
import numpy as np 
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt
# %%
data_x, data_y = load_veterans_lung_cancer()

# %%


time, survival_prob, conf_int = kaplan_meier_estimator(
    data_y["Status"], data_y["Survival_in_days"], conf_type="log-log"
)
plt.step(time, survival_prob, where="post")
plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
plt.ylim(0, 1)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
# %%

for treatment_type in ['standard', 'test']:
    mask_treatment = data_x['Treatment'] == treatment_type
    time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
    data_y["Status"][mask_treatment], data_y["Survival_in_days"][mask_treatment], conf_type="log-log")

    plt.step(time_treatment, survival_prob_treatment, where='post', label=f"Treatment: {treatment_type}")
    plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step='post')
plt.ylim(0, 1)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")   

# %%
for value in data_x['Celltype'].unique():
    mask = data_x['Celltype'] == value 
    time_cell, survival_prob_cell, conf_int = kaplan_meier_estimator(
        data_y['Status'][mask], data_y['Survival_in_days'][mask], conf_type='log-log'
    )
    plt.step(time_cell, survival_prob_cell, where='post', label=f"Celltype: {value} (n = {mask.sum()})")
    plt.fill_between(time_cell, conf_int[0], conf_int[1], alpha=0.25, step='post')
plt.ylim(0, 1)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")   
# %%


data_x_numeric = OneHotEncoder().fit_transform(data_x)

# %%
estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x_numeric, data_y)
# %%
pd.Series(estimator.coef_, index=data_x_numeric.columns).sort_values(ascending=False)

# %%
x_new = pd.DataFrame.from_dict(
    {
        1: [65, 0, 0, 1, 60, 1, 0, 1],
        2: [65, 0, 0, 1, 60, 1, 0, 0],
        3: [65, 0, 1, 0, 60, 1, 0, 0],
        4: [65, 0, 1, 0, 60, 1, 0, 1],
    },
    columns=data_x_numeric.columns,
    orient="index",
)
# %%
pred_surv = estimator.predict_survival_function(x_new)
time_points = np.arange(1, 1000)
for i, surv_func in enumerate(pred_surv):
    plt.step(time_points, surv_func(time_points), where="post", label=f"Sample {i + 1}")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
# %%
from sksurv.metrics import concordance_index_censored

prediction = estimator.predict(data_x_numeric)
result = concordance_index_censored(data_y["Status"], data_y["Survival_in_days"], prediction)
# %%
########################################
## Which variable is most predictive? ##
########################################
#%%
import numpy as np

n_features = data_x_numeric.shape[1]
scores = np.empty(n_features)
#%%
def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

#%%
scores = fit_and_score_features(data_x_numeric.values, data_y)
pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False)
# %%
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    [
        ("encode", OneHotEncoder()),
        ("select", SelectKBest(fit_and_score_features, k=3)),
        ("model", CoxPHSurvivalAnalysis()),
    ]
)
#%%
from sklearn.model_selection import GridSearchCV, KFold
#%%
param_grid = {"select__k": np.arange(1, data_x_numeric.shape[1] + 1)}
cv = KFold(n_splits=3, random_state=1, shuffle=True)
gcv = GridSearchCV(pipe, param_grid, return_train_score=False, cv=cv)
gcv.fit(data_x, data_y)
#%%
results = pd.DataFrame(gcv.cv_results_).sort_values(by="mean_test_score", ascending=False)
results.loc[:, ~results.columns.str.endswith("_time")]
# %%
pipe.set_params(**gcv.best_params_)
pipe.fit(data_x, data_y)

encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()])
# %%
