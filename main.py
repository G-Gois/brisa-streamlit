import streamlit as st
import numpy as np
import pandas as pd
from classes.graphs import Graphs
import pickle
from classes.sample_data import SampleData

def on_form_submit():
    print("Form submitted")


with st.sidebar:
    st.title("Configurações do modelo")
    st.subheader("Teste")
    with st.form("config"):
        st.write("Modelo:")
        selected_model = st.selectbox("Modelo:",["Random Forest"])
        data = st.date_input("Data do pedido:")
        is_feriado = st.checkbox("É feriado?")
        chuva = st.text_input("Chuva:",placeholder="Previsão de mm para o dia")

        submitted = st.form_submit_button("Executar",on_click=on_form_submit)

predictions = pd.read_csv("./data/{}_predictions.csv".format(selected_model.replace(" ","")))
temp = pd.read_csv("./data/data_set_temp.csv")
graphs = Graphs()

@st.cache(persist=True)
def load_model():
    with open('models/{}.pkl'.format(selected_model.replace(" ","")), 'rb') as f:
        model = pickle.load(f)
    return model

sample = SampleData()


with st.container():
    st.header("Modelo: " + selected_model)
    st.divider()
    st.subheader("Histórico de pedidos")
    st.line_chart(temp,x="data",y="quantidade_pro",height=0,use_container_width=True,color=(220,100,0))
    st.divider()
    st.subheader("Previsão X Histórico")
    pred_vs_hist = pd.DataFrame({"Previsão":predictions["Predicted"],"Histórico":temp["quantidade_pro"]})
    st.line_chart(predictions,y="Actual",x="Predicted",height=0,use_container_width=True)
    st.line_chart(temp,x="data",y="quantidade_pro",height=0,use_container_width=True)

# Gráficos
# Confusion matrix
# Desvio padrao