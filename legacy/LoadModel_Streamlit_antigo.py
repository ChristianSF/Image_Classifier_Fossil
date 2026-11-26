import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import keras

modelo = tf.keras.models.load_model("/content/drive/MyDrive/Fossil_Classification/Modelo_Novo.h5")

batch_size = 32
img_height = 180
img_width = 180

classes = ['Arthropoda', 'Bryozoa', 'Mollusca']

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Selecione um arquivo', filenames)
    return os.path.join(folder_path, selected_filename)

if __name__ == '__main__':

    espaco = ""

    st.title('Classificador de filos por imagens de fósseis (Arthropoda, Bryozoa, ou Mollusca)')
    
    st.write(espaco)

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

    
    st.header("Teste o modelo: ")
    my_expander = st.expander(label='Clique para saber mais')
    my_expander.write("""Para testar o modelo, basta fazer o upload de uma imagem de algum dos três filos, 
    [Arthropoda](https://science.mnhn.fr/institution/mnhn/collection/f/item/list?phylum=Arthropoda), 
    [Bryozoa](https://science.mnhn.fr/institution/mnhn/collection/f/item/list?phylum=Bryozoa), ou 
    [Mollusca](https://science.mnhn.fr/institution/mnhn/collection/f/item/list?phylum=Mollusca).
     Estas imagens podem ser encontradas no site: [Museu de História Natural da França](https://www.mnhn.fr/fr).""")

    st.write(espaco)

    st.subheader("Escolha uma imagem: ")
    if st.checkbox('Selecione algum arquivo da sua pasta atual:'):
        folder_path = '.'
        if st.checkbox('Mudar de pasta'):
            folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        st.write('Você selecionou: `%s`' % filename)

        img = keras.preprocessing.image.load_img(
        filename, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 

        predictions = modelo.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        st.write("Sua imagem é: ")
        st.image(filename, use_column_width=False)

        st.subheader("Esse fóssil pertence ao filo {} com {:.2f}% de precisão.".format(classes[np.argmax(score)], 100 * np.max(score)))
