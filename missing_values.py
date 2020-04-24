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

# we will use only numerical predictors

X = X_full.select_dtypes(exclude='object')
X_test = X_test_full.select_dtypes(exclude='object')

#%%

# Splitting the data for validations

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)
#%%

# Checking the first five rows
print (X.head())

#%%

# Primary Investigations

# Shape of the training data (rows, columns)
print (X_train.shape)

# Number of misssing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# number of rows 1168
# number of missing column 3
# number of missing data 212 + 6 + 58

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

# DROPING null rows from data

# drop columns in training data ( This is not good, x_valid and X_train might not
# have the same na we should do it according to x_train)

#reduced_X_train = X_train.dropna(axis=1)
#reduced_X_valid = X_valid.dropna(axis=1)


# Another way

col_with_missing = [ col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.copy().drop(col_with_missing, axis=1)
reduced_X_valid = X_valid.copy().drop(col_with_missing, axis=1)

#%%

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

#%%

from sklearn.impute import SimpleImputer

# Impultion part A

# save imputer without any params
imputer = SimpleImputer()

# fitting data and transforming (fit only works for list I think)
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

# Imputed data dosent have column data
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

#%%

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

#%%

# Lets see if imputation works with missing column
col_with_missing = [ col for col in X_train.columns if X_train[col].isnull().any()]
X_train_to_imputed = X_train.copy()
X_valid_to_imputed = X_valid.copy()
for missing_col in col_with_missing:
    X_train_to_imputed[missing_col + "_missing"] = X_train[missing_col].isnull()
    X_valid_to_imputed[missing_col + "_missing"] = X_valid[missing_col].isnull()

# fitting data and transforming (fit only works for list I think)
imputed_X_train_with_missing = pd.DataFrame(imputer.fit_transform(X_train_to_imputed))
imputed_X_valid_with_missing = pd.DataFrame(imputer.transform(X_valid_to_imputed))

# Imputed data dosent have column data
imputed_X_train_with_missing.columns = X_train_to_imputed.columns
imputed_X_valid_with_missing.columns = X_valid_to_imputed.columns

print("MAE (Imputation with missing column):")
print(score_dataset(imputed_X_train_with_missing, imputed_X_valid_with_missing, y_train, y_valid))

#%%
# Lets see if imputation works with missing column
col_with_missing = [ col for col in X_train.columns if X_train[col].isnull().any()]
final_imputer = SimpleImputer(strategy='median')
X_train_with_zero = X_train.copy()
X_valid_with_zero = X_valid.copy()

X_train_with_zero['GarageYrBlt'] = X_train_with_zero['GarageYrBlt'].fillna(0)
X_valid_with_zero['GarageYrBlt'] = X_valid_with_zero['GarageYrBlt'].fillna(0)

for missing_col in col_with_missing:
    X_train_with_zero[missing_col + "_missing"] = X_train[missing_col].isnull()
    X_valid_with_zero[missing_col + "_missing"] = X_valid[missing_col].isnull()

# fitting data and transforming (fit only works for list I think)
imputed_X_train_with_zero = pd.DataFrame(final_imputer.fit_transform(X_train_with_zero))
imputed_X_valid_with_zero = pd.DataFrame(final_imputer.transform(X_valid_with_zero))

# Imputed data dosent have column data
imputed_X_train_with_zero.columns = X_train_with_zero.columns
imputed_X_valid_with_zero.columns = X_valid_with_zero.columns

print("MAE (Imputation with missing column with zero):")
print(score_dataset(imputed_X_train_with_zero, imputed_X_valid_with_zero, y_train, y_valid))