import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

dados = pd.read_csv(r"Dados.csv")

#images = [Image.open(''), Image.open('img/Img_00104.jpg'), Image.open('img/Img_00154.jpg')]
#st.image(images, use_column_width=True, caption=["some generic text"] * len(images), width=10)

espaco = ""

st.title('Uso de ferramentas de aprendizagem de máquina e processamento de imagens para gestão de coleções de fósseis')

st.markdown(''' Reconhecimento de imagens, processamento de linguagem natural e
diversas outras tarefas que antes pareciam impossíveis de serem realizadas de
forma autônoma por computadores e dispositivos eletrônicos, agora, são
extremamente comuns e acessíveis por meio das técnicas de aprendizado profundo
e inteligência artificial. Tendo isso em vista, este trabalho tem como objetivo propor
um ferramenta que auxilie museus na organização e identificação de fósseis, por
meio de uma rede neural convolucional escrita na linguagem de programação
Python, que é uma das linguagens mais utilizadas para esse tipo de tarefa.
 ''')
st.write(espaco)
st.markdown("Dados e fotos foram coletados no site do [Museu de História Natural da França](https://www.mnhn.fr/fr).")

st.header("Algumas das fotos coletadas:")
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.image("img/Img_0049.jpg")
    st.write("Exemplo de legenda")
with col2:
    st.image("img/Img_00104.jpg")
    st.write("Exemplo de legenda")
with col3:
    st.image("img/Img_00274.jpg")
    st.write("Exemplo de legenda")

col4, col5, col6 = st.beta_columns(3)
with col1:
    st.image("img/Img_00201.jpg")
    st.write("Exemplo de legenda")
with col2:
    st.image("img/Img_00235.jpg")
    st.write("Exemplo de legenda")
with col3:
    st.image("img/Img_00266.jpg")
    st.write("Exemplo de legenda")

st.header('EDA - Análise Exploratória de Dados')

st.text(espaco)
st.markdown("Verficando o quantidade de registros no dataset, utilizando **dados.shape** ")
st.code(dados.shape, language='python')

st.markdown("Visualizando a quantidade de colunas, utlizando **dados.columns** ")
st.code(dados.columns, language='python')

st.write("Exibindo as 5 primeiras linhas do conjunto de dados, com **dados.head():** ", dados.head())
st.text(espaco)

st.write("Exibindo as 5 ultimas linhas do conjunto de dados, com **dados.tail():** ", dados.tail())

st.subheader("Analisando quantidade de dados faltantes e duplicados")
st.markdown("Soma de NaN em cada coluna, utilizando **dados.isnull().sum()** ")
st.code(dados.isnull().sum(), language='python')
st.text("Podemos ver que não há nenhum NaN neste dataset")

st.text(espaco)
st.markdown("Soma de valores duplicados no dataset, utilizando **dados.duplicated().sum()** ")
st.code(dados.duplicated().sum(), language='python')

st.subheader("Analisando os  presentes no conjunto de dados")
st.markdown("Analisandos a quantidade de fósseis presentes em cada filo, utilizando: **dados['filo'].value_counts()**")
st.code(dados['filo'].value_counts(), language='python')

st.markdown("Visualizando a porcentagem de fósseis presentes em cada filo, utilizando: **dados['filo'].value_counts(normalize=True)**")
st.code(dados['filo'].value_counts(normalize=True)*100, language='python')

layout = go.Layout(title="Número de Fósseis por filo presentes no conjunto de dados", xaxis= {'title':'Filos'},
                   yaxis=dict(title='N° de fósseis'), hovermode='closest')
fig = go.Figure(data=[go.Histogram(x=dados['filo'], marker_color='#330C73')], layout=layout)

st.plotly_chart(fig, use_container_width=True)

st.markdown("Como podemos ver pelo gráfico, o Filo **Mollusca**, é o filo com maior quantidade de fósseis "
            "presentes em nosso conjunto, por isso, iremos separar este filo, e analisar a classe biológica de"
            "cada filo. Para separar os dados, utilizaremos o código abaixo:")

st.code("molusca = dados[dados['filo'] == 'Mollusca']", language='python')

molusca = dados[dados['filo'] == 'Mollusca']

layout = go.Layout(title="Número de Fósseis por Classe pertecentes ao Filo 'Mollusca' no dataset", xaxis= {'title':'Classes'},
                   yaxis=dict(title='N° de fósseis'), hovermode='closest')
fig2 = go.Figure(data=[go.Histogram(x=molusca['classe'],marker_color='#EB89B5')], layout=layout)

st.plotly_chart(fig2, use_container_width=True)


