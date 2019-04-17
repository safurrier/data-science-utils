#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
SEED = 42
np.random.seed(SEED)



# ## Load Data

df = pd.read_csv('model_data.csv')


# ## Params


MODEL = xgb.XGBClassifier
# MODEL = xgb.XGBRegressor
SCORING_METHOD = 'roc_auc' # Must be compatible with GridSearchCV
CV=5


# # Split Data

X = df.drop(columns='target')
y = df['target']

# Use if needed
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ## Grid Search

# In[13]:


grid =  {
    'n_estimators': [100],
    'max_depth': range(2, 12, 2),
    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    'subsample': np.arange(0.1, .91, 0.1),
    'min_child_weight': range(2, 20, 2), 
    }
gsc = GridSearchCV(
    estimator=MODEL,
    param_grid=grid,
    cv=CV, 
    scoring=SCORING_METHOD, 
    verbose=1, 
    njobs=-1)
grid_result = gsc.fit(X, y)

# In[18]:


grid_result.best_params_


# In[19]:


grid_result.best_score_


# In[22]:


grid_result.best_params_


# ## Tree Search (Number of estimators to use)

# In[36]:


# put best params in a list
trees_grid = {key:[value]for key, value in grid_result.best_params_.items()}
# Add trees
trees_grid['n_estimators'] = [1, 10, 100, 300, 500, 1000, 2500]
trees_grid    


# In[ ]:


trees_gsc = GridSearchCV(
    estimator=MODEL,
    param_grid=trees_grid,
    cv=CV, 
    scoring=SCORING_METHOD, 
    verbose=1)
trees_grid_result = gsc.fit(X, y)

# In[47]:


trees_grid_result.best_params_


# In[48]:


trees_grid_result.best_score_


# ## 10 Fold CV Score

# In[55]:


np.mean(cross_val_score(MODEL(**trees_grid_result.best_params_), X, y, scoring=SCORING_METHOD, n_jobs=-1, cv=10))



