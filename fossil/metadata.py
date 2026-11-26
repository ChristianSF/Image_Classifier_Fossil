"""Leitura e limpeza dos metadados coletados do MNHN (coleta de 2021).

Problema conhecido do crawler antigo: quando a página de um espécime não tinha
imagem, a variável `foto` mantinha o valor do espécime anterior. Por isso há
sequências de linhas consecutivas apontando para a mesma foto. Apenas a primeira
linha de cada sequência é de fato dona da imagem; as demais não têm foto.
"""

from pathlib import Path

import pandas as pd

VALORES_AUSENTES = ["Não informado", "Não_informado", "Não informada", ""]

COLUNAS_TEXTO = [
    "colecao_original", "estado_colecao", "filo", "classe", "ordem", "familia",
    "genero", "especie", "nome", "país", "estado_provincia", "municipio",
    "nome_coletor", "era", "sistema", "series", "estagio",
]


def carregar_metadados(caminho: str | Path) -> pd.DataFrame:
    """Lê o CSV bruto, remove espaços extras e padroniza valores ausentes."""
    df = pd.read_csv(caminho)
    for col in COLUNAS_TEXTO:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(VALORES_AUSENTES, pd.NA)
    return df


def marcar_fotos_validas(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna `foto_valida`.

    Uma foto é válida apenas na primeira linha de uma sequência de linhas
    consecutivas com a mesma URL. As repetições seguintes são herança do bug
    do crawler e recebem foto_valida = False, com `foto` e `nome_foto` zerados.
    """
    df = df.copy()
    inicio_sequencia = df["foto"] != df["foto"].shift()
    df["foto_valida"] = inicio_sequencia & df["foto"].notna()
    df.loc[~df["foto_valida"], ["foto", "nome_foto"]] = pd.NA
    return df


def limpar_metadados(caminho: str | Path) -> pd.DataFrame:
    return marcar_fotos_validas(carregar_metadados(caminho))
