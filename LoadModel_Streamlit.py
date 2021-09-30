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
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

if __name__ == '__main__':
    # Select a file
    if st.checkbox('Select a file in current directory'):
        folder_path = '.'
        if st.checkbox('Change directory'):
            folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)

        img = keras.preprocessing.image.load_img(
        filename, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 

        predictions = modelo.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        st.write("Essa classe pertence a classe {} com {:.2f}% de precisão.".format(classes[np.argmax(score)], 100 * np.max(score)))

