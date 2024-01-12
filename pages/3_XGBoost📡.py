import streamlit as st
from classes.graphs import Graphs
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="SVR",
    page_icon="📈",
    layout="wide",
)

graphs = Graphs()

st.header("XGBoost")
st.write("O XGBoost é uma biblioteca de aprendizado de máquina que implementa o algoritmo Gradient Boosting, sendo conhecido por seu desempenho e eficácia em problemas de classificação e regressão. Ele utiliza uma abordagem de ensemble learning e oferece recursos avançados para lidar com dados desbalanceados e outros desafios.")
st.write('O problema encontrado neste modelo foi que ele é capaz apenas de produzir um resultado por vez, logo, temos três modelos diferentes em operação')
st.divider()

st.header("Resultados:")

st.header("Desempenho do modelo:")
st.write("Para medição de desempenho do modelo, foi utilizado o Erro Percentual Absoluto Médio (MAPE)")
st.latex(r"""MAPE = 100 \times \frac{1}{n} \sum_{i=1}^n \frac{|y_t - y_p|}{y_t}""")

def calcMAPE(y_true,y_pred):
    errors = abs(y_pred - y_true)
    mape = 100 * (errors / y_true)
    accuracy = round(100 - np.mean(mape),2)

    return accuracy

delivery_accuracy = calcMAPE(graphs.y['qnt_delivery'],graphs.y_delivery_XG['Delivery'])
local_accuracy = calcMAPE(graphs.y['qnt_local'],graphs.y_local_XG['Local'])
pickup_accuracy = calcMAPE(graphs.y['qnt_pickup'],graphs.y_pickup_XG['Pickup'])

df = {"Retorno": ["Delivery","Local","Retirada"],"MAPE":[str(delivery_accuracy)+'%',str(local_accuracy)+'%',str(pickup_accuracy)+'%']}
st.table(df)

graphs.delivery(graphs.y_delivery_XG)
graphs.pickup(graphs.y_pickup_XG)
graphs.local(graphs.y_local_XG)
