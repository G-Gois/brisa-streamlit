import streamlit as st
from classes.store import Store
from classes.graphs import Graphs
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Random Forest",
    page_icon="🎄",
    layout="wide",
)

store = Store()
graphs = Graphs()

st.header("Random Forest")
st.write(" Random Forest é um modelo versátil e poderoso que combina várias árvores de decisão para realizar previsões ou classificações. Sua aleatoriedade e combinação de previsões de várias árvores ajudam a reduzir o sobreajuste e aumentar a robustez do modelo.")
st.write("O modelo com o melhor desempenho foi aquele com 128 árvores na floresta.")
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

delivery_accuracy = calcMAPE(graphs.y['qnt_delivery'],graphs.y_predRF['Delivery'])
local_accuracy = calcMAPE(graphs.y['qnt_local'],graphs.y_predRF['Local'])
pickup_accuracy = calcMAPE(graphs.y['qnt_pickup'],graphs.y_predRF['Pickup'])

df = {"Retorno": ["Delivery","Local","Retirada"],"MAPE":[str(delivery_accuracy)+'%',str(local_accuracy)+'%',str(pickup_accuracy)+'%']}
st.table(df)


graphs.delivery(graphs.y_predRF)
graphs.local(graphs.y_predRF)
graphs.pickup(graphs.y_predRF)
