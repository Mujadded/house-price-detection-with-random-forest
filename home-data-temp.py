#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
#%%

# Read the data
X_full = pd.read_csv('./input/Melbourne_housing_FULL.csv')

#X_full = pd.read_csv('./input/melbourne_from_kaggle_full.csv')
#print(X_full.columns)

#%%
from sklearn.impute import SimpleImputer
needed_features = ['Rooms', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Propertycount', 'Price']
# Imputation
X_taking = X_full[needed_features]
cols_with_missing = [col for col in X_taking.columns
                     if X_taking[col].isnull().any()]

for col in cols_with_missing:
    X_taking[col + '_was_missing'] = X_taking[col].isnull()
    
my_imputer = SimpleImputer()
imputed_X_full = pd.DataFrame(my_imputer.fit_transform(X_taking))

imputed_X_full.columns = X_taking.columns

#%%
y = imputed_X_full.Price

features = needed_features = ['Rooms', 'Bedroom2', 'Bedroom2_was_missing',
                   'Bathroom_was_missing','Car_was_missing',
                   'Landsize_was_missing', 'BuildingArea_was_missing',
                   'Propertycount_was_missing', 'Price_was_missing',
                   'Bathroom', 'Car', 'Landsize', 'BuildingArea',
                   'Propertycount']
X = imputed_X_full[features]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

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