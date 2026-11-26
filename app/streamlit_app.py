"""App Streamlit: análise dos dados coletados e classificador de fósseis.

Rodar a partir da raiz do repositório:
    streamlit run app/streamlit_app.py

O modelo é lido da pasta indicada em MODELO_DIR (padrão: modelos/filo_cnn),
que deve conter modelo.keras e classes.json (gerados por scripts/treinar_baseline.py).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from fossil.metadata import limpar_metadados  # noqa: E402

CAMINHO_DADOS = RAIZ / "data" / "dados_mnhn_2021.csv"
MODELO_DIR = Path(os.environ.get("MODELO_DIR", RAIZ / "modelos" / "filo_cnn"))
TAMANHO_IMG = (180, 180)

EXEMPLOS = ["Img_0049.jpg", "Img_00104.jpg", "Img_00274.jpg", "Img_00201.jpg", "Img_00235.jpg", "Img_00266.jpg"]


@st.cache_data
def carregar_dados():
    return limpar_metadados(CAMINHO_DADOS)


@st.cache_resource
def carregar_modelo(pasta: Path):
    import keras  # importado aqui para a aba de dados abrir sem TensorFlow

    modelo = keras.models.load_model(pasta / "modelo.keras")
    classes = json.loads((pasta / "classes.json").read_text())
    return modelo, classes


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def aba_dados():
    df = carregar_dados()
    com_foto = df[df["foto_valida"]]

    st.markdown(
        "Dados e fotos coletados em 2021 do acervo de fósseis do "
        "[Muséum national d'Histoire naturelle](https://science.mnhn.fr/institution/mnhn/collection/f/item/search) (Paris)."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Espécimes", f"{len(df):,}".replace(",", "."))
    c2.metric("Com foto válida", f"{len(com_foto):,}".replace(",", "."))
    c3.metric("Filos", df["filo"].nunique())

    st.info(
        "O crawler original repetia a foto do espécime anterior quando a página não tinha imagem. "
        "Essas linhas foram marcadas como sem foto (`foto_valida = False`)."
    )

    st.subheader("Exemplos de imagens")
    for linha in (EXEMPLOS[:3], EXEMPLOS[3:]):
        for col, nome in zip(st.columns(3), linha):
            col.image(str(RAIZ / "img" / nome), use_container_width=True)

    st.subheader("Espécimes por filo")
    apenas_com_foto = st.checkbox("Considerar apenas espécimes com foto válida", value=True)
    base = com_foto if apenas_com_foto else df
    contagem = base["filo"].fillna("Não informado").value_counts().reset_index()
    contagem.columns = ["filo", "n"]
    st.plotly_chart(px.bar(contagem, x="filo", y="n", labels={"n": "Nº de espécimes", "filo": "Filo"}), use_container_width=True)

    st.subheader("Classes dentro de um filo")
    filos = contagem["filo"].tolist()
    filo = st.selectbox("Filo", filos, index=filos.index("Mollusca") if "Mollusca" in filos else 0)
    classes = base.loc[base["filo"] == filo, "classe"].fillna("Não informado").value_counts().reset_index()
    classes.columns = ["classe", "n"]
    st.plotly_chart(px.bar(classes, x="classe", y="n", labels={"n": "Nº de espécimes", "classe": "Classe"}), use_container_width=True)

    st.subheader("Espécimes por era geológica")
    eras = base["era"].fillna("Não informado").value_counts().reset_index()
    eras.columns = ["era", "n"]
    st.plotly_chart(px.bar(eras, x="era", y="n", labels={"n": "Nº de espécimes", "era": "Era"}), use_container_width=True)

    with st.expander("Ver tabela"):
        st.dataframe(base)


def aba_classificador():
    if not (MODELO_DIR / "modelo.keras").exists():
        st.warning(
            f"Modelo não encontrado em `{MODELO_DIR}`. Treine com `scripts/treinar_baseline.py` "
            "ou defina a variável de ambiente MODELO_DIR."
        )
        return

    modelo, classes = carregar_modelo(MODELO_DIR)
    st.markdown(f"Classes do modelo: **{', '.join(classes)}**")

    arquivo = st.file_uploader("Envie a foto de um fóssil", type=["jpg", "jpeg", "png"])
    if arquivo is None:
        return

    img = Image.open(arquivo).convert("RGB")
    st.image(img, width=300)

    entrada = np.asarray(img.resize(TAMANHO_IMG), dtype="float32")[None, ...]
    probs = softmax(modelo.predict(entrada, verbose=0)[0])
    ordem = np.argsort(probs)[::-1]

    st.subheader(f"Previsão: {classes[ordem[0]]}")
    st.caption("Confiança é a probabilidade do softmax, não uma medida de acerto do modelo.")
    for i in ordem:
        st.progress(float(probs[i]), text=f"{classes[i]}: {100 * probs[i]:.1f}%")


def main():
    st.set_page_config(page_title="Classificador de fósseis", layout="wide")
    st.title("Classificação de fósseis por imagem")
    st.markdown(
        "Ferramenta para apoiar museus na organização e identificação de fósseis, "
        "usando redes neurais convolucionais."
    )
    dados, classificador = st.tabs(["Dados", "Classificador"])
    with dados:
        aba_dados()
    with classificador:
        aba_classificador()


if __name__ == "__main__":
    main()
