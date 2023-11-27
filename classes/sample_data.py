import pandas as pd
import numpy as np

class SampleData():
    def __init__(self):
        pass

    def get_sales_data(self):
        dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='7D')

        sales_data = [] 
        is_predicted = [False] * len(dates_2022)
        for i in range(len(dates_2022)):
            if i % 2 ==0:
                is_predicted[i] = True
            sales_data.append(np.random.randint(0, 1000))

        df = pd.DataFrame({'Date': dates_2022, 'Sales': sales_data, 'Predicted': is_predicted})
        return df
