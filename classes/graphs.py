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
        self.y_predRF = pd.read_csv('data/predictionsRF.csv',parse_dates=['data'])
        
        self.y_delivery_SVR = pd.read_csv('data/predictionsSVRDelivery.csv',parse_dates=['data'])
        self.y_pickup_SVR = pd.read_csv('data/predictionsSVRPickup.csv',parse_dates=['data'])
        self.y_local_SVR = pd.read_csv('data/predictionsSVRLocal.csv',parse_dates=['data'])

        self.data2022 = pd.read_csv('data/training_data2022.csv',parse_dates=['data'])
        self.y = pd.read_csv('data/test_data2023.csv',parse_dates=['data'])
        self.all_data = pd.concat([self.data2022, self.y])

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

    def delivery(self,pred) -> None:
         
        new_df = pred

        # Sort the datetime index
        new_df['data'] = sorted(new_df['data'])
        # Plot
        fig = plt.figure(figsize=(10, 6))
        plt.title('Previsão de delivery')
        plt.plot(self.all_data['data'], self.all_data['qnt_delivery'], label='Real', alpha=0.7)
        plt.plot(new_df['data'], new_df['Delivery'], label='Previsto')
        plt.xlabel('Data')
        plt.ylabel('Quantidade de delivery')
        plt.legend()

        fig_html = mpld3.fig_to_html(fig)

        # Display the plot in Streamlit
        components.html(fig_html,height=700,scrolling=True)
    def local(self,pred) -> None:
        new_df = pred

        # Sort the datetime index
        new_df['data'] = sorted(new_df['data'])

        # Plot
        fig = plt.figure(figsize=(10, 6))
        plt.title('Previsão de consumo local')
        plt.plot(self.all_data['data'], self.all_data['qnt_local'], label='Real', alpha=0.7)
        plt.plot(new_df['data'], new_df['Local'], label='Previsto')
        plt.xlabel('Data')
        plt.ylabel('Quantidade de consumo local')
        plt.legend()

        fig_html = mpld3.fig_to_html(fig)

        components.html(fig_html,height=700,scrolling=True)
    def pickup(self,pred) -> None:
        new_df = pred

        # Sort the datetime index
        new_df['data'] = sorted(new_df['data'])

        # Plot
        fig = plt.figure(figsize=(10, 6))
        plt.title('Previsão de retirada')
        plt.plot(self.all_data['data'], self.all_data['qnt_pickup'], label='Real', alpha=0.7)
        plt.plot(new_df['data'], new_df['Pickup'], label='Previsto')
        plt.xlabel('Data')
        plt.ylabel('Quantidade de retirada')
        plt.legend()

        fig_html = mpld3.fig_to_html(fig)

        # Display the plot in Streamlit
        components.html(fig_html,height=700,scrolling=True)
