"""Lista as imagens, remove duplicatas e gera o split treino/val/teste.

Uso:
    python scripts/preparar_splits.py --imagens /caminho/Coleta_30-06-2021/Teste --saida data/splits_filo.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fossil.splits import dividir, listar_imagens, remover_duplicatas  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imagens", required=True, help="Pasta com uma subpasta por classe")
    p.add_argument("--saida", required=True)
    p.add_argument("--frac-val", type=float, default=0.15)
    p.add_argument("--frac-teste", type=float, default=0.15)
    p.add_argument("--min-por-classe", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = listar_imagens(args.imagens)
    print(f"Imagens encontradas: {len(df)}")
    df, info = remover_duplicatas(df)
    print(f"Removidas por rótulo ambíguo: {info['removidas_rotulo_ambiguo']}")
    print(f"Removidas por serem cópias: {info['removidas_copias']}")

    df, descartadas = dividir(df, args.frac_val, args.frac_teste, args.min_por_classe, args.seed)
    if descartadas:
        print(f"Classes descartadas (menos de {args.min_por_classe} imagens): {descartadas}")

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False)
    print("\nImagens por classe e split:")
    print(df.pivot_table(index="rotulo", columns="split", values="caminho", aggfunc="count").to_string())


if __name__ == "__main__":
    main()
