import streamlit as st
import pandas as pd
from classes.graphs import Graphs
from classes.sample_data import SampleData
from classes.store import Store

store = Store()
graphs = Graphs()
MODELS = ["Random Forest","SVR"]

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

with st.sidebar:
    st.title("Autores:")
    st.write("Eduardo Bedin")
    st.write("Gabriel Weber")
    st.write("Rafael Hentz")
    st.write("Welyton Leidens")

with st.container():
    st.header("Modelo: " + store.model_name)
    st.divider()
    st.subheader("Amostra de dados:")
    st.write("Os dados utilizados são do ano de 2022 e 2023. As informações de 2022 foram usadas para treino do modelo em uma granularidade diária, enquanto os dados de 2023 são desconhecidos pelo modelo.")
    st.write("Estrutura dos dados:")
    st.dataframe(store.all_data.drop('Data',axis=1).head())

    st.divider()
    st.header("Resultados:")
    graphs.timeseries(store)
    st.divider()
    st.header("Desempenho do modelo:")
    st.write("Para medição de desempenho do modelo, foi utilizado o Erro Percentual Absoluto Médio (MAPE)")
    st.latex(r"""MAPE = 100 \times \frac{1}{n} \sum_{i=1}^n \frac{|y_t - y_p|}{y_t}""")
    graphs.desempenho(store)
    csv = pd.read_csv('data/predictions.csv',parse_dates=['data']).set_index('data')
    graphs.delivery()
    graphs.local()
    graphs.pickup()
