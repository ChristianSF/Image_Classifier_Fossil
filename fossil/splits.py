"""Montagem do conjunto de imagens e divisão estratificada em treino, validação e teste.

Estrutura esperada das imagens (a mesma gerada pelo crawler antigo):

    raiz/<classe>/*.jpg

onde <classe> é o rótulo (ex: Arthropoda, Bryozoa, Mollusca ou as classes de
Mollusca). Cada arquivo corresponde a um espécime, então a divisão por arquivo
já é uma divisão por espécime.
"""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EXTENSOES = {".jpg", ".jpeg", ".png"}


def _md5(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def listar_imagens(raiz: str | Path) -> pd.DataFrame:
    raiz = Path(raiz)
    linhas = [
        {"caminho": str(p), "rotulo": p.parent.name}
        for p in sorted(raiz.rglob("*"))
        if p.suffix.lower() in EXTENSOES and p.parent != raiz
    ]
    if not linhas:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {raiz}")
    return pd.DataFrame(linhas)


def remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove imagens idênticas (mesmo md5).

    Se a mesma imagem aparece com rótulos diferentes, todas as cópias saem,
    porque o rótulo é ambíguo (caso típico de imagem placeholder do site).
    """
    df = df.copy()
    df["md5"] = df["caminho"].map(lambda c: _md5(Path(c)))
    rotulos_por_hash = df.groupby("md5")["rotulo"].nunique()
    ambiguos = rotulos_por_hash[rotulos_por_hash > 1].index
    n_ambiguos = int(df["md5"].isin(ambiguos).sum())
    df = df[~df["md5"].isin(ambiguos)]
    n_antes = len(df)
    df = df.drop_duplicates("md5")
    info = {"removidas_rotulo_ambiguo": n_ambiguos, "removidas_copias": n_antes - len(df)}
    return df.reset_index(drop=True), info


def dividir(
    df: pd.DataFrame,
    frac_val: float = 0.15,
    frac_teste: float = 0.15,
    min_por_classe: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Divisão estratificada. Classes com menos de `min_por_classe` imagens saem."""
    contagem = df["rotulo"].value_counts()
    descartadas = sorted(contagem[contagem < min_por_classe].index)
    df = df[~df["rotulo"].isin(descartadas)].reset_index(drop=True)

    treino_val, teste = train_test_split(
        df, test_size=frac_teste, stratify=df["rotulo"], random_state=seed
    )
    frac_val_relativa = frac_val / (1 - frac_teste)
    treino, val = train_test_split(
        treino_val, test_size=frac_val_relativa, stratify=treino_val["rotulo"], random_state=seed
    )
    treino, val, teste = treino.assign(split="treino"), val.assign(split="val"), teste.assign(split="teste")
    return pd.concat([treino, val, teste]).reset_index(drop=True), descartadas
