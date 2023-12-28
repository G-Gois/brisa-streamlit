import streamlit as st
import pandas as pd
from classes.graphs import Graphs
from classes.sample_data import SampleData
from classes.store import Store

store = Store()
graphs = Graphs()

def on_form_submit():
    store.load_model(selected_model)
    X, y = store.X,store.y

    predictions = store.model.predict(X)
    result = pd.DataFrame({"Predicted":predictions,"Actual":y})
    store.result = result

with st.sidebar:
    st.title("Configurações do modelo")
    with st.form("config"):
        selected_model = st.selectbox("Modelo:",["Random Forest","SVR"])
        submitted = st.form_submit_button("Executar",on_click=on_form_submit)

with st.container():
    st.header("Modelo: " + store.model_name)
    st.divider()
    st.subheader("Resultados:")
    graphs.timeseries(store)
    st.divider()
    st.subheader("Desempenho do modelo:")
