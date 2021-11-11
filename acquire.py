import pandas as pd
from env import host, user, password
import os
def get_url(db):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    if os.path.isfile('titanic.csv'):
        return pd.read_csv('titanic.csv', index_col=0)
    else:
        titanic = pd.read_sql('SELECT * FROM passengers', get_url('titanic_db'))
        titanic.to_csv('titanic.csv')
        return titanic

def get_iris_data():
    if os.path.isfile('iris.csv'):
        return pd.read_csv('iris.csv', index_col=0)
    else:
        iris = pd.read_sql('SELECT * FROM measurements JOIN species USING(species_id)', get_url('iris_db'))
        iris.to_csv('iris.csv')
        return iris

def get_telco_data():
    if os.path.isfile('telco.csv'):
        return pd.read_csv('telco.csv', index_col=0)
    else:
        sql = '''
        SELECT *
        FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN internet_service_types USING(internet_service_type_id)
        JOIN payment_types USING(payment_type_id)
        '''
        telco = pd.read_sql(sql, get_url('telco_churn'))
        telco.to_csv('telco.csv')
        return telco