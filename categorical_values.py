# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('./input/melbourne_from_kaggle_full.csv')
X_test_full = pd.read_csv('./input/melbourne_from_kaggle_test.csv')


#%%
# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

#%%

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X_full.columns if X_full[col].isnull().any()] 
X_full.drop(cols_with_missing, axis=1, inplace=True)
X_test_full.drop(cols_with_missing, axis=1, inplace=True)
#%%

# Splitting the data for validations

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)
#%%

from sklearn.preprocessing import OneHotEncoder

string_tuple = (X_train.dtypes == 'object')

object_cols = list(string_tuple[string_tuple].index)

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(oh_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(oh_encoder.transform(X_valid[object_cols]))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

X_train_oh = pd.concat([num_X_train, OH_cols_train], axis=1)
X_valid_oh = pd.concat([num_X_valid, OH_cols_valid], axis=1 )


#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function to comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


#%%

print("MAE (Drop columns with missing values):")
print(score_dataset(X_train_oh, X_valid_oh, y_train, y_valid))
