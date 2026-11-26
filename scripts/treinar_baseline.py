"""Treina e avalia a CNN baseline a partir do CSV de splits.

Uso:
    python scripts/treinar_baseline.py --splits data/splits_filo.csv --saida modelos/filo_cnn

Saídas em --saida:
    modelo.keras, classes.json, metricas.json (inclui AUC-ROC), relatorio.txt,
    matriz_confusao.csv, curvas_roc.csv, historico.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fossil.model import TAMANHO_IMG, cnn_baseline  # noqa: E402


def carregar_imagem(caminho, rotulo):
    img = tf.io.decode_image(tf.io.read_file(caminho), channels=3, expand_animations=False)
    img = tf.image.resize(img, TAMANHO_IMG)
    return img, rotulo


def criar_dataset(df, classes, batch, embaralhar, seed):
    idx = {c: i for i, c in enumerate(classes)}
    ds = tf.data.Dataset.from_tensor_slices((df["caminho"].values, df["rotulo"].map(idx).values))
    if embaralhar:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(carregar_imagem, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def calcular_auc(y_true: np.ndarray, probs: np.ndarray, classes: list[str]):
    """AUC-ROC one-vs-rest.

    Retorna as métricas agregadas (macro e ponderada), a AUC por classe e os
    pontos da curva ROC de cada classe, para plotar depois.
    """
    n = len(classes)
    if n == 2:
        auc = float(roc_auc_score(y_true, probs[:, 1]))
        agregadas = {"auc_roc": auc}
    else:
        agregadas = {
            "auc_roc_macro": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
            "auc_roc_ponderada": float(roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")),
        }

    por_classe, curvas = {}, []
    for i, classe in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.min() == y_bin.max():  # classe ausente no teste
            por_classe[classe] = None
            continue
        por_classe[classe] = float(roc_auc_score(y_bin, probs[:, i]))
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        curvas.append(pd.DataFrame({"classe": classe, "fpr": fpr, "tpr": tpr}))

    return agregadas, por_classe, pd.concat(curvas, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--epocas", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sem-augmentation", action="store_true", help="Reproduz o setup de 2021")
    p.add_argument("--sem-peso-classes", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    keras.utils.set_random_seed(args.seed)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.splits)
    classes = sorted(df["rotulo"].unique())
    treino, val, teste = (df[df["split"] == s] for s in ("treino", "val", "teste"))

    ds_treino = criar_dataset(treino, classes, args.batch, True, args.seed)
    ds_val = criar_dataset(val, classes, args.batch, False, args.seed)
    ds_teste = criar_dataset(teste, classes, args.batch, False, args.seed)

    pesos = None
    if not args.sem_peso_classes:
        y = treino["rotulo"].map({c: i for i, c in enumerate(classes)}).values
        valores = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
        pesos = dict(enumerate(valores))

    modelo = cnn_baseline(len(classes), augmentation=not args.sem_augmentation)
    modelo.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(saida / "modelo.keras", monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(saida / "historico.csv"),
    ]
    modelo.fit(ds_treino, validation_data=ds_val, epochs=args.epocas, class_weight=pesos, callbacks=callbacks)

    # Avaliação no teste (nunca visto no treino nem na seleção do modelo)
    logits = modelo.predict(ds_teste)
    y_pred = logits.argmax(axis=1)
    y_true = teste["rotulo"].map({c: i for i, c in enumerate(classes)}).values
    probs = softmax(logits)
    auc_agregadas, auc_por_classe, curvas_roc = calcular_auc(y_true, probs, classes)

    metricas = {
        **auc_agregadas,
        "auc_roc_por_classe": auc_por_classe,
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_ponderado": float(f1_score(y_true, y_pred, average="weighted")),
        "n_teste": int(len(y_true)),
        "classes": classes,
        "augmentation": not args.sem_augmentation,
        "peso_classes": not args.sem_peso_classes,
        "seed": args.seed,
    }
    relatorio = classification_report(y_true, y_pred, target_names=classes, digits=4)
    matriz = pd.DataFrame(confusion_matrix(y_true, y_pred), index=classes, columns=classes)

    (saida / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    (saida / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2))
    (saida / "relatorio.txt").write_text(relatorio)
    matriz.to_csv(saida / "matriz_confusao.csv")
    curvas_roc.to_csv(saida / "curvas_roc.csv", index=False)

    print(relatorio)
    print("AUC-ROC:", json.dumps({**auc_agregadas, **auc_por_classe}, indent=2))
    print(matriz.to_string())


if __name__ == "__main__":
    main()
