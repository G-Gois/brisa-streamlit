import streamlit as st
from classes.store import Store
from classes.graphs import Graphs
import pandas as pd

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
graphs.timeseries(store)

col1, col2 = st.columns(2)
with col1:
    graphs.pickup()

with col2:
    graphs.local()

st.divider()
st.header("Desempenho do modelo:")
st.write("Para medição de desempenho do modelo, foi utilizado o Erro Percentual Absoluto Médio (MAPE)")
st.latex(r"""MAPE = 100 \times \frac{1}{n} \sum_{i=1}^n \frac{|y_t - y_p|}{y_t}""")
df = {"Retorno": ["Delivery","Local","Retirada"],"MAPE":[100,100,100]}
st.table(df)
graphs.desempenho(store)
