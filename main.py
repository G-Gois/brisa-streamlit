import streamlit as st
import numpy as np
import pandas as pd

def on_form_submit():
    print("Form submitted")
    st.write(data)
    st.write(is_feriado)
    st.write(chuva)


chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['Sales', 'Rain', 'Taxes'])

with st.container():
    st.header("Previsão de vendas")
    st.line_chart(chart_data)   


with st.sidebar:
    st.title("Configurações do modelo")
    with st.form("config"):
        data = st.date_input("Data do pedido:")
        is_feriado = st.checkbox("É feriado?")
        chuva = st.text_input("Chuva:",placeholder="Previsão de mm para o dia")
        # submit button
        submitted = st.form_submit_button("Executar",on_click=on_form_submit)


