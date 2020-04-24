#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
#%%

# Read the data
X_full_raw = pd.read_csv('./input/Melbourne_housing_FULL.csv')

X_full_raw.dropna(axis=0, subset=['Price'], inplace=True)
y = X_full_raw.Price
X_full_raw.drop(['Price'],axis=1,inplace=True)

#%%

X = X_full_raw.select_dtypes(exclude='object')

#%%

X_missing_cols= (X.isnull().sum())
print(X_missing_cols[X_missing_cols > 0])

#%%
from sklearn.impute import SimpleImputer


cols_with_missing = [col for col in X.columns
                     if X[col].isnull().any()]

for col in cols_with_missing:
    X[col + '_was_missing'] = X[col].isnull()
    
my_imputer = SimpleImputer(strategy='mean')
imputed_X_full = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X_full.columns = X.columns

#%%

X_train, X_valid, y_train, y_valid = train_test_split(imputed_X_full, y, train_size=0.8, test_size=0.2, random_state=0)

#%%

from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_4, model_5]



#%%
from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


#%%
for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
    
# Model 1 MAE: 178647
# Model 2 MAE: 177648
# Model 3 MAE: 180804
# Model 4 MAE: 223035