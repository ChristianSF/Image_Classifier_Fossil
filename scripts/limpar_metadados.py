"""Gera a versão limpa dos metadados de 2021.

Uso:
    python scripts/limpar_metadados.py --entrada data/dados_mnhn_2021.csv --saida data/dados_mnhn_2021_limpo.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fossil.metadata import limpar_metadados  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", default="data/dados_mnhn_2021.csv")
    p.add_argument("--saida", default="data/dados_mnhn_2021_limpo.csv")
    args = p.parse_args()

    df = limpar_metadados(args.entrada)
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False)

    n = len(df)
    n_validas = int(df["foto_valida"].sum())
    print(f"Espécimes: {n}")
    print(f"Com foto válida: {n_validas}")
    print(f"Foto herdada do espécime anterior (bug do crawler): {n - n_validas}")
    print("\nEspécimes com foto válida por filo:")
    print(df.loc[df["foto_valida"], "filo"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
