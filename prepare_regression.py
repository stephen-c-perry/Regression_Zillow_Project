import numpy as np
import pandas as pd
from scipy import stats
import sklearn.preprocessing as p
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wrangle as w
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import TweedieRegressor




#minmaxscaler
def scale_zillow(train, validate, test):

    train_1 = train.copy()
    validate_1 = validate.copy()
    test_1 = test.copy()

    scale_cols = ['bedrooms', 'bathrooms', 'total_sqft', 'year_built']
    minmax_scaler = p.MinMaxScaler()
    minmax_scaler.fit(train_1[scale_cols])

    train_1[scale_cols] = minmax_scaler.transform(train[scale_cols])
    validate_1[scale_cols] = minmax_scaler.transform(validate[scale_cols])
    test_1[scale_cols] = minmax_scaler.transform(test[scale_cols])

    df_train_1 = pd.DataFrame(train_1).set_index([train_1.index.values])
    df_validate_1 = pd.DataFrame(validate_1).set_index([validate_1.index.values])
    df_test_1 = pd.DataFrame(test_1).set_index([test_1.index.values])

    return df_train_1, df_validate_1, df_test_1




def scale_dataframes(df1, df2, df3):
    scaler = MinMaxScaler()
    df1_scaled = scaler.fit_transform(df1)
    df2_scaled = scaler.transform(df2)
    df3_scaled = scaler.transform(df3)
    return pd.DataFrame(df1_scaled), pd.DataFrame(df2_scaled), pd.DataFrame(df3_scaled)

'''
def GLM(power, alpha):
    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.tax_value)

    # predict train
    y_train['value_pred_lm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = e.rmse(y_train.tax_value, y_train.y_pred_mean)

    # predict validate
    y_validate['value_pred_lm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = e.rmse(y_validate.tax_value, y_validate.y_pred_median)

    return print("RMSE for GLM using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

'''
