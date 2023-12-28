import streamlit as st
import pandas as pd
from classes.graphs import Graphs
from classes.sample_data import SampleData
from classes.store import Store

store = Store()
graphs = Graphs()
MODELS = ["Random Forest","SVR"]
def on_form_submit():
    store.load_model(selected_model)

with st.sidebar:
    st.title("Configurações do modelo")
    with st.form("config"):
        selected_model = st.selectbox("Modelo:",MODELS)
        submitted = st.form_submit_button("Executar",on_click=on_form_submit)
        st.caption("Clique em 'Executar' para iniciar o modelo.")
    st.title("Autores:")
    st.write("Eduardo Bedin")
    st.write("Gabriel Weber")
    st.write("Rafael Hentz")
    st.write("Welyton Leidens")

with st.container():
    st.header("Modelo: " + store.model_name)
    st.divider()
    st.subheader("Resultados:")
    graphs.timeseries(store)
    st.divider()
    st.subheader("Desempenho do modelo:")
    graphs.desempenho(store)
