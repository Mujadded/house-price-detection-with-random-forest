#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:21:03 2020

@author: mujadded
"""

#LODING DATA

import pandas as pd

X_raw = pd.read_csv('./input/melbourne_from_kaggle_full.csv')
X_test_raw = pd.read_csv('./input/test.csv', index_col="Id")

#%%

# DATABASE ANALYSIS
missin_val_count_by_column = (X_raw.isnull().sum())
print(missin_val_count_by_column[missin_val_count_by_column > 0])

print('For test')
# DATABASE ANALYSIS
missin_val_count_by_column = (X_test_raw.isnull().sum())
print(missin_val_count_by_column[missin_val_count_by_column > 0])

# IMPUTED NEEDED
# LotFrontage
# 

# Missing Value replace
# Alley, FireplaceQu, Fence, MiscFeature -> replace with "none"
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 -> replace with 'N/A'
# Electrical, GarageCond, GarageQual, GarageFinish, GarageType, PoolQC -> "N/A"
# GarageYrBlt -> 0


# DELETE ROW WITH NAN
# MasVnrType

#%%


# Delete row with nan
X_raw.dropna(axis=0, subset=['MasVnrType'], inplace=True)

#%%
# y
y = X_raw.SalePrice
X_raw.drop(['SalePrice'], axis=1,inplace=True)

#%%
# Replacing Values

replace_with_zero = ['GarageYrBlt']
X_raw[replace_with_zero] = X_raw[replace_with_zero].fillna(0)
X_test_raw[replace_with_zero] = X_test_raw[replace_with_zero].fillna(0)


replace_with_none = ['Alley', 'FireplaceQu', 'Fence', 'MiscFeature']
X_raw[replace_with_none] = X_raw[replace_with_none].fillna('none')
X_test_raw[replace_with_none] = X_test_raw[replace_with_none].fillna('none')

replace_with_na = ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                   'BsmtFinType2', 'Electrical', 'FireplaceQu', 
                   'GarageType','GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']
X_raw[replace_with_na] = X_raw[replace_with_na].fillna('N/A')
X_test_raw[replace_with_na] = X_test_raw[replace_with_na].fillna('N/A')

#%%
# Impute value

from sklearn.impute import SimpleImputer

X_num = X_raw.select_dtypes(exclude='object')
X_num_test = X_test_raw.select_dtypes(exclude='object')

cols_with_missing = [col for col in X_num.columns
                     if X_num[col].isnull().any()]

for col in cols_with_missing:
    X_num[col + '_was_missing'] = X_num[col].isnull()
    X_num_test[col + '_was_missing'] = X_num_test[col].isnull()
    
imputer = SimpleImputer(strategy='median')
X_num_imputed = pd.DataFrame(imputer.fit_transform(X_num))
X_num_test_imputed = pd.DataFrame(imputer.transform(X_num_test))


X_num_imputed.columns = X_num.columns
X_num_test_imputed.columns= X_num_test.columns


X_num_imputed.index = X_num.index
X_num_test_imputed.index = X_num_test.index
#%%

# Hotloadencoding

from sklearn.preprocessing import OneHotEncoder

X_string= X_raw.select_dtypes(include='object')
X_test_string= X_test_raw.select_dtypes(include='object')
X_test_string = X_test_string.fillna('N/A')
#object_cols = list(X_string.columns)
#object_test_cols = list(X_test_string.columns)

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(oh_encoder.fit_transform(X_string))
OH_test_cols = pd.DataFrame(oh_encoder.transform(X_test_string))

OH_cols.index = X_string.index
OH_test_cols.index = X_test_string.index

#%%
# Final dataset

X = pd.concat([OH_cols,X_num_imputed], axis=1)

X_test = pd.concat([OH_test_cols, X_num_test_imputed], axis=1)
#%%

from sklearn.model_selection import train_test_split

# Splitting the data for validations

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)

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
print(score_dataset(X_train, X_valid, y_train, y_valid))

#15965.149347079041
#%%
#model = RandomForestRegressor(n_estimators=100, random_state=0)
#model.fit(X, y)
#preds_test = model.predict(X_test)
#output = pd.DataFrame({'Id': X_test.index,
#                       'SalePrice': preds_test})
#output.to_csv('submission.csv', index=False)