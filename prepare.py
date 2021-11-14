import pandas as pd
import numpy as np
import os
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import 

# visualize
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('figure', figsize=(8, 6))
plt.rc('font', size=13)

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

# acquire
from env import host, user, password
from pydataset import data

import acquire

#-------------------------------------
def prep_iris(df):
    df = df.drop(columns =['species_id'])
    df = df.rename(columns= {'species_name' : 'species'})
    dummy_df = pd.get_dummies(df['species'], drop_first = True)
    df = pd.concat([df, dummy_df], axis = 1)
    
    return df

#-------------------------------------------
def prep_titanic(df):
    df = df.drop_duplicates()
    df = df.drop(columns = ['embarked', 'passenger_id'])
    df_dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first = False)
    df = pd.concat([df, df_dummies], axis=1)
    
    return df

#---------------------------------------------------
def prep_telco(telco_df):
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id']
    telco_df = telco_df.drop(columns=cols_to_drop)
    telco_df['total_charges'] = pd.to_numeric(telco_df['total_charges'], errors = 'coerce')
    telco_df['total_charges'] = telco_df['total_charges'].fillna(0)
    telco_df = telco_df[telco_df['total_charges'] != 0]
    telco_dummy_df = pd.get_dummies(telco_df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'contract_type', 'internet_service_type', 'payment_type'  ]], dummy_na = False, drop_first = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
    telco_df = pd.concat([telco_df, telco_dummy_df], axis = 1)
    return telco_df