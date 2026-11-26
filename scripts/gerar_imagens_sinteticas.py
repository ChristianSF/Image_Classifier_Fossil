"""Gera um conjunto pequeno de imagens sintéticas para testar o pipeline.

Cria a mesma estrutura esperada pelos dados reais (uma subpasta por classe),
para rodar preparar_splits.py e treinar_baseline.py de ponta a ponta sem as
imagens do museu. Cada classe tem uma forma e uma cor dominante, com ruído e
posição aleatórios, então a CNN deve aprender a separá-las em poucas épocas.

Uso:
    python scripts/gerar_imagens_sinteticas.py --saida data/sinteticas --por-classe 40
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FORMAS = ["elipse", "retangulo", "triangulo", "linhas", "cruz"]
CORES = [(200, 80, 60), (60, 160, 90), (70, 90, 200), (190, 170, 60), (150, 70, 170)]


def desenhar(forma: str, cor: tuple, tamanho: int, rng: np.random.Generator) -> Image.Image:
    fundo = rng.integers(0, 60, size=(tamanho, tamanho, 3), dtype=np.uint8)
    img = Image.fromarray(fundo)
    d = ImageDraw.Draw(img)
    m = tamanho // 6
    x0, y0 = (int(v) for v in rng.integers(0, 2 * m, size=2))
    x1, y1 = tamanho - 1 - int(rng.integers(0, 2 * m)), tamanho - 1 - int(rng.integers(0, 2 * m))
    cor = tuple(int(np.clip(c + rng.integers(-25, 26), 0, 255)) for c in cor)
    if forma == "elipse":
        d.ellipse([x0, y0, x1, y1], fill=cor)
    elif forma == "retangulo":
        d.rectangle([x0, y0, x1, y1], fill=cor)
    elif forma == "triangulo":
        d.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=cor)
    elif forma == "linhas":
        for y in range(y0, y1, max(4, tamanho // 12)):
            d.line([x0, y, x1, y], fill=cor, width=3)
    else:
        cx, cy, w = (x0 + x1) // 2, (y0 + y1) // 2, tamanho // 10
        d.rectangle([cx - w, y0, cx + w, y1], fill=cor)
        d.rectangle([x0, cy - w, x1, cy + w], fill=cor)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saida", default="data/sinteticas")
    p.add_argument("--classes", nargs="+", default=["Arthropoda", "Bryozoa", "Mollusca"])
    p.add_argument("--por-classe", type=int, default=40)
    p.add_argument("--tamanho", type=int, default=220)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if len(args.classes) > len(FORMAS):
        p.error(f"No máximo {len(FORMAS)} classes")

    rng = np.random.default_rng(args.seed)
    for i, classe in enumerate(args.classes):
        pasta = Path(args.saida) / classe
        pasta.mkdir(parents=True, exist_ok=True)
        for j in range(args.por_classe):
            img = desenhar(FORMAS[i], CORES[i], args.tamanho, rng)
            img.save(pasta / f"sint_{j:04d}.jpg", quality=90)
    total = len(args.classes) * args.por_classe
    print(f"{total} imagens em {args.saida} ({args.por_classe} por classe)")


if __name__ == "__main__":
    main()
