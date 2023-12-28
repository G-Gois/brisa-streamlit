import streamlit as st
import pandas as pd
from .sample_data import SampleData
from .store import Store
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import streamlit.components.v1 as components


def display_data(cursor):
    x = cursor.artist.get_xdata()[cursor.target.index]
    y = cursor.artist.get_ydata()[cursor.target.index]
    label = f"X: {x}\nY: {y}"
    cursor.annotation.set_text(label)
    cursor.annotation.get_bbox_patch().set(fc='white', ec='black', lw=1, alpha=0.9)

class Graphs:
    def __init__(self) -> None:
        self.sample_data = SampleData().get_sales_data()
        self.store = Store()

    def timeseries(self,store:Store):

        X, y = store.X,store.y
        new_df = pd.concat([store.test_data, store.previous_data],ignore_index=False)


        X.drop('Data', axis=1, inplace=True)

        pred = store.model.predict(new_df['2023':].drop('qnt_delivery',axis=1).drop('Data',axis=1))
        residuals = y - pred
        # Compute the standard deviation of the residuals
        pred_std = np.std(residuals)

        # Compute the confidence interval manually
        pred_ci = pd.DataFrame({'lower': pred - 1.96 * pred_std, 'upper': pred + 1.96 * pred_std})
        plot_data = pd.DataFrame({'data':new_df['2023':].index,'value':pred})
        # Plotting
        new_df = new_df.sort_index()
        fig,ax = plt.subplots(figsize=(10, 6))
        plt.plot(new_df.index, new_df['qnt_delivery'], label='Real')
        plt.plot(new_df['2023':].index, plot_data['value'], label='Previsto', alpha=0.7)
        plt.fill_between(new_df['2023':].index, pred_ci['lower'], pred_ci['upper'], color='k', alpha=0.2)
        plt.xlabel('Data')
        plt.ylabel('Quantidade de delivery')
        plt.legend()

        fig_html = mpld3.fig_to_html(fig)

        # Display the plot in Streamlit
        components.html(fig_html,height=700,scrolling=True)
