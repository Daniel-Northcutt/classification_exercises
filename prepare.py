import pandas as pd
import numpy as np
import os
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

### 
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


#---------------------------------------------------
def clean_data(df):
    '''
    This function will clean the data etc etc...'''
    
    df = df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class']
    df = df.drop(columns = cols_to_drop)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    df['baseline_prediction'] = 0
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True,True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df

#---------------------------------------------------
def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test sbusert dataframes
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    return train, validate, test

#---------------------------------------------------
def impute_mode(train, validate, test):
    '''
    takes in train, validate, and test and uses train to identify the gbset value to replace nulls in embark_town
    imputes that value in to  all three sets and returns all three sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imputer = imputer.fit(train[['embark_town']])
    train[['embark_town']] = imputer.transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

def prep_titanic_data(df):
    '''
    the ultimate dishwasher
    '''
    df = clean_data(df)
    train, validate, test = split_data(df)
    return train, validate, test


def prep_telco(df):
    df = df.drop_duplicates()
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    df = df.drop(columns = cols_to_drop)
    df.total_charges = df.total_charges.replace(' ',0)
    df.total_charges = df.total_charges.astype(float)
    cols_to_dummy = df[['gender','partner','dependents','phone_service','multiple_lines',
                        'online_security','online_backup','device_protection','tech_support','streaming_tv',
                        'streaming_movies','paperless_billing','churn','contract_type',
                        'internet_service_type','payment_type']]
    dummy_df = pd.get_dummies(cols_to_dummy, dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis = 1)

    return df

def split_telco_data(df):
    '''
    Takes in a dataframe and returns train, validate, test sbusert dataframes
    '''
    telco_train, telco_test = train_test_split(df, test_size = .2, stratify=df.churn_Yes)
    telco_train, telco_validate = train_test_split(telco_train, test_size=.3, stratify=telco_train.churn_Yes)
    return telco_train, telco_validate, telco_test
