import pickle
import pandas as pd
class Store:
    _instance = None
    model_name = 'Random Forest'
    model = None
    test_data = None
    previous_data = None
    y_predicted = None
    def __init__(self):
        self.load_test_data()
        self.load_prev_data()
        self.all_data =  pd.concat([self.test_data, self.previous_data],ignore_index=False)

    def __new__(self, *args, **kwargs):
        if not self._instance:
            self._instance = super().__new__(self, *args, **kwargs)
        return self._instance
    

    def load_all_data(self):
        return pd.concat([self.previous_data, self.test_data[0]])

    def load_model(self,model:str):
        self.model_name = model
        with open('models/{}.pkl'.format(model.replace(" ","")), 'rb') as f:
            loaded_model = pickle.load(f)
            self.model = loaded_model
    
    def load_prev_data(self):
        input_csv = pd.read_csv('data/training_data2022.csv',parse_dates=['data'])
        orders = input_csv.set_index('data')

        column_order = ['data','mm','temp_mean','temp_max','temp_min','feriado','qnt_delivery','qnt_pickup','qnt_local']
        orders = orders.reindex(columns=column_order)
        
        orders['day_of_week'] = orders.index.dayofweek
        orders['month'] = orders.index.month
        orders['year'] = orders.index.year

        for x in orders:
            if orders[x].dtypes == "int64":
                orders[x] = orders[x].astype(float)

        without_objects = orders.select_dtypes(exclude=['object'])
        dropped = without_objects.drop('qnt_pickup',axis=1)
        dropped = dropped.drop('qnt_local',axis=1)
        dropped = dropped.rename(columns={'data':'Data'})

        self.previous_data = dropped
    
    def load_test_data(self):
        input_csv = pd.read_csv('data/test_data2023.csv',parse_dates=['data'])
        orders = input_csv.set_index('data')

        column_order = ['data','mm','temp_mean','temp_max','temp_min','feriado','qnt_delivery','qnt_pickup','qnt_local']
        orders = orders.reindex(columns=column_order)
        
        orders['day_of_week'] = orders.index.dayofweek
        orders['month'] = orders.index.month
        orders['year'] = orders.index.year

        for x in orders:
            if orders[x].dtypes == "int64":
                orders[x] = orders[x].astype(float)

        without_objects = orders.select_dtypes(exclude=['object'])
        dropped = without_objects.drop('qnt_pickup',axis=1)
        dropped = dropped.drop('qnt_local',axis=1)
        dropped = dropped.rename(columns={'data':'Data'})
        y = dropped['qnt_delivery']
        X = dropped.drop('qnt_delivery',axis=1)
        self.X = X
        self.y = y

        self.test_data = dropped