# -*- coding: utf-8 -*-
"""Teste_Load_Model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15Zdgucbh5ZH8l5ywbjipGJ-LURjQzCNr
"""

!pip install dash

!pip install jupyter-dash

!pip install dash-bootstrap-components

import base64
from io import BytesIO

import numpy as np

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import jupyter_dash
from PIL import Image, ImageDraw

import tensorflow as tf
import tensorflow_hub as hub
import keras

modelo = tf.keras.models.load_model("/content/drive/MyDrive/Fossil_Classification/Modelo_Novo.h5")

modelo.summary()

batch_size = 32
img_height = 180
img_width = 180

classes = ['Arthropoda', 'Bryozoa', 'Mollusca']

teste = "/content/Mollusca_Amphineura_Img_106.jpg"


img = keras.preprocessing.image.load_img(
      teste, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = modelo.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("Essa classe pertence a classe {} com {:.2f}% de precisão.".format(classes[np.argmax(score)], 100 * np.max(score)))

"""#Novo"""

def image_card(src, header=None):
    return dbc.Card(
        [
            dbc.CardHeader(header),
            dbc.CardBody(html.Img(src=src, style={"width": "100%"})),
        ]
    )

def preprocess_b64(image_enc):
    """Preprocess b64 string into TF tensor"""
    decoded = base64.b64decode(image_enc.split("base64,")[-1])
    hr_image = tf.image.decode_image(decoded)

    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]

    return tf.expand_dims(tf.cast(hr_image, tf.float32), 0)


def tf_to_b64(tensor, ext="jpeg"):
    buffer = BytesIO()

    image = tf.cast(tf.clip_by_value(tensor[0], 0, 255), tf.uint8).numpy()
    Image.fromarray(image).save(buffer, format=ext)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{ext};base64, {encoded}"

app = jupyter_dash.JupyterDash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = [
    dcc.Upload(
        dbc.Card(
            "Drag and Drop or Click",
            body=True,
            style={
                "textAlign": "center",
                "borderStyle": "dashed",
                "borderColor": "black",
            },
        ),
        id="img-upload",
        multiple=False,
    )
]


app.layout = dbc.Container(
    [
        html.H1("Image Fossil Classifier - TensorFlow, Plotly and Dash"),
        html.Hr(),
        dbc.Row([dbc.Col(c) for c in controls]),
        html.Br(),
        dbc.Spinner(
            dbc.Row(
                [
                    dbc.Col(html.Div(id=img_id))
                    for img_id in ["original-img", "enhanced-img"]
                ]
            )
        )
    ],
    fluid=True,
)

@app.callback(
    [Output("original-img", "children"), Output("enhanced-img", "children")],
    [Input("img-upload", "contents")],
    [State("img-upload", "filename")],
)
def enhance_image(img_str, filename):
    if img_str is None:
        return dash.no_update, dash.no_update

    # sr_str = img_str # PLACEHOLDER
    low_res = preprocess_b64(img_str)
    #super_res = model(tf.cast(low_res, tf.float32))
    predictions = model.predict(low_res)
    score = tf.nn.softmax(predictions[0])
    img_teste = Image.new('RGB', (420, 100), color = (73, 109, 137))
 
    d = ImageDraw.Draw(img_teste)
    d.text((10,10), "Essa classe pertence a classe {} com {:.2f}% de precisão.".format(classes[np.argmax(score)], 100 * np.max(score)), fill=(255,255,0))
    img_teste.save('teste.png')

    lr = image_card(img_str, header="Original Image")
    sr = image_card(sr_str, header="Enhanced Image")

    return lr, sr

app.run_server(mode='external')

predictions = modelo.predict(teste)
score = tf.nn.softmax(predictions[0])
img_teste = Image.new('RGB', (420, 100), color = (73, 109, 137))
 
d = ImageDraw.Draw(img_teste)
d.text((10,10), "Essa classe pertence a classe {} com {:.2f}% de precisão.".format(classes[np.argmax(score)], 100 * np.max(score)), fill=(255,255,0))
img_teste.save('teste.png')
sr_str = tf_to_b64('teste.png')

img = keras.preprocessing.image.load_img(
      teste, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = modelo.predict(img_array)
score = tf.nn.softmax(predictions[0])

#print("Essa classe pertence a classe {} com {:.2f}% de precisão.".format(classes[np.argmax(score)], 100 * np.max(score)))

img_teste = Image.new('RGB', (420, 100), color = (73, 109, 137))
 
d = ImageDraw.Draw(img_teste)
d.text((10,10), "Essa classe pertence a classe {} com {:.2f}% de precisão.".format(classes[np.argmax(score)], 100 * np.max(score)), fill=(255,255,0))
img_teste.save('teste.png')

