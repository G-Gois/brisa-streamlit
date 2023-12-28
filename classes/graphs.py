from sklearn import metrics
import pandas as pd
from .sample_data import SampleData
from .store import Store
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import streamlit.components.v1 as components
import streamlit as st
class Graphs:
    def __init__(self) -> None:
        self.sample_data = SampleData().get_sales_data()
        self.store = Store()

    def timeseries(self,store:Store) -> None:
        new_df = store.all_data

        # Get predictions
        pred = store.model.predict(new_df['2023':].drop('qnt_delivery',axis=1).drop('Data',axis=1))
        store.y_predicted = pred
        # Get confidence intervals
        residuals = store.y - pred
        pred_std = np.std(residuals)
        pred_ci = pd.DataFrame({'lower': pred - 1.96 * pred_std, 'upper': pred + 1.96 * pred_std})
        
        # Sort the datetime index
        new_df = new_df.sort_index()

        # Plot
        fig = plt.figure(figsize=(10, 6))
        plt.title('Previsão de delivery')
        plt.plot(new_df.index, new_df['qnt_delivery'], label='Real')
        plt.plot(new_df['2023':].index, pred, label='Previsto', alpha=0.7)
        plt.fill_between(new_df['2023':].index, pred_ci['lower'], pred_ci['upper'], color='k', alpha=0.2)
        plt.xlabel('Data')
        plt.ylabel('Quantidade de delivery')
        plt.legend()

        fig_html = mpld3.fig_to_html(fig)

        # Display the plot in Streamlit
        components.html(fig_html,height=700,scrolling=True)
    
    def desempenho(self,store:Store) -> None:
        fig = plt.figure(figsize=(10,6))

        plt.scatter(store.y, store.y_predicted, c='crimson')

        p1 = max(max(store.y_predicted), max(store.y))
        p2 = min(min(store.y_predicted), min(store.y))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')

        fig_html = mpld3.fig_to_html(fig)

        errors = abs(store.y_predicted - store.y)

        mape = 100 * (errors / store.y)
        accuracy = 100 - np.mean(mape)
        st.write('Precisão:', round(accuracy, 2), '%.')
        st.write('Erro Médio Absoluto:', round(np.mean(errors), 2), 'pedidos.')
        components.html(fig_html,height=700,scrolling=True)

