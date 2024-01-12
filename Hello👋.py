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


with st.container():
    st.title("Hello, World!👋")
    st.divider()
    
    st.subheader("Projeto:")
    st.write("O projeto teve seu início com um desafio: construir um modelo de inteligência artificial capaz de produzir recomendações de produtos. \n \
             Contudo, o projeto seguiu um rumo diferente.")
    st.write("Em conversa com a empresa e orientadores, decidimos optar por algo mais palpável e necessário para a empresa: Um sistema de predição de pedidos.")
    st.divider()
    
    st.subheader("Desafios:")
    st.write("Os maiores desafios encontrados foram relacionados à falta de experiência no desenvolvimento de modelos inteligentes utilizando Python.")
    st.write("Para superar isso, contamos com a ajuda do Prof. Orientador Jacson, onde nos apresentou uma série de cursos e nos orientou durante a aprendizagem.")
    st.divider()
    
    st.subheader("Amostra de dados:")
    st.write("Com base nos dados fornecidos pela Amo, criamos uma série de metadados, como temperatura, pluviosidade, dia da semana e feriados. O objetivo era entender como tudo isso se relacionava.")
    st.write("Os dados utilizados são do ano de 2022 e 2023. As informações de 2022 foram usadas para treino do modelo em uma granularidade diária, enquanto os dados de 2023 são desconhecidos pelo modelo.")
    st.write("Estrutura dos dados:")
    st.dataframe(store.all_data.drop('Data',axis=1).head())
    
    st.subheader("Autores:")
    st.write("Eduardo Bedin | Gabriel Weber | Rafael Hentz | Welyton Leidens")
 
