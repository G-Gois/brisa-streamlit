import streamlit as st
import numpy as np
import pandas as pd

def on_form_submit():
    print("Form submitted")



with st.sidebar:
    st.title("Configurações do modelo")
    with st.form("config"):
        st.write("Modelo:")
        selected_model = st.selectbox("Modelo:",["Random Forest"])
        data = st.date_input("Data do pedido:")
        is_feriado = st.checkbox("É feriado?")
        chuva = st.text_input("Chuva:",placeholder="Previsão de mm para o dia")

        submitted = st.form_submit_button("Executar",on_click=on_form_submit)

predictions = pd.read_csv("./data/{}_predictions.csv".format(selected_model.replace(" ","")))

with st.container():
    st.header("Modelo: " + selected_model)
    st.subheader("Previsão de vendas")
    st.subheader("Precisão do modelo")
    predictions.drop("Unnamed: 0",axis=1,inplace=True)
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(predictions,x="Actual",y="Predicted")
    with col2:
        st.dataframe(predictions)
