# Classificação de fósseis por imagem

Classificação de fósseis a partir de fotografias, com o objetivo de apoiar museus na organização e identificação de acervos. O projeto começou em 2021 como iniciação científica, com dados e imagens da [coleção de fósseis do Muséum national d'Histoire naturelle](https://science.mnhn.fr/institution/mnhn/collection/f/item/search) (Paris), e treinou redes neurais convolucionais para dois problemas:

- **Filo**: Arthropoda, Bryozoa e Mollusca.
- **Classe dentro de Mollusca**: Amphineura, Bivalvia, Cephalopoda, Gastropoda e Scaphopoda.

## Estrutura

```
app/streamlit_app.py        App com análise dos dados e classificador
fossil/                     Código reutilizável
  metadata.py               Leitura e limpeza dos metadados
  splits.py                 Listagem de imagens, deduplicação e split treino/val/teste
  model.py                  CNN baseline
scripts/
  limpar_metadados.py       Gera o CSV limpo
  preparar_splits.py        Gera o CSV de splits a partir da pasta de imagens
  treinar_baseline.py       Treina e avalia a CNN no conjunto de teste
data/dados_mnhn_2021.csv    Metadados coletados em 2021 (9.489 espécimes)
img/                        Imagens de exemplo usadas no app
notebooks/                  Notebooks originais de 2021 (EDA e treinos)
legacy/                     Código original de 2021 (crawler e apps), mantido como histórico
```

## Como usar

```bash
pip install -r requirements.txt

# 1. Metadados limpos
python scripts/limpar_metadados.py

# 2. Split estratificado (a pasta deve ter uma subpasta por classe)
python scripts/preparar_splits.py --imagens /caminho/para/imagens --saida data/splits_filo.csv

# 3. Treino e avaliação
python scripts/treinar_baseline.py --splits data/splits_filo.csv --saida modelos/filo_cnn

# 4. App
MODELO_DIR=modelos/filo_cnn streamlit run app/streamlit_app.py
```

Para reproduzir o setup de 2021 (sem augmentation e sem peso por classe):

```bash
python scripts/treinar_baseline.py --splits data/splits_filo.csv --saida modelos/filo_cnn_2021 \
    --sem-augmentation --sem-peso-classes --epocas 10
```

## O que mudou em relação a 2021

**Correção nos dados.** O crawler original mantinha a foto do espécime anterior quando a página não tinha imagem. Com isso, 2.981 dos 9.489 registros apontavam para a foto de outro espécime (uma mesma foto chegava a aparecer em 727 linhas seguidas). Depois da correção, restam 6.508 espécimes com foto válida. As contagens por filo mudam bastante: Foraminifera cai de 1.717 para 502 e Arthropoda de 982 para 496.

**Avaliação.** Os notebooks usavam só treino e validação (80/20), e a validação também servia para acompanhar o treino. Agora há um conjunto de teste separado, split estratificado, remoção de imagens duplicadas (inclusive a mesma imagem com rótulos diferentes) e métricas por classe: F1 macro, AUC-ROC (one-vs-rest, macro, ponderada e por classe), relatório por classe e matriz de confusão. Isso é importante porque as classes são muito desbalanceadas.

**Treino.** A arquitetura da CNN é a mesma, com data augmentation, dropout, peso por classe, early stopping e checkpoint do melhor modelo. Os resultados de 2021 (cerca de 89% de acurácia na validação, com 99% no treino) indicavam overfitting.

**App.** Os dois apps viraram um só, com upload de imagem no lugar do seletor de arquivos locais, o modelo lido de uma pasta configurável e não mais de um caminho do Google Drive. O texto "precisão" foi trocado por "confiança", que é o que o softmax representa.

## Próximos passos

- Substituir o crawler por dados do [GBIF](https://www.gbif.org/dataset/b275a4c1-9859-4f3c-8ead-d86dde820fbc), onde a coleção do MNHN é publicada com API oficial, e incluir outras coleções paleontológicas.
- Comparar a CNN baseline com transfer learning (EfficientNet, ConvNeXt), embeddings de modelos de fundação (DINOv2, CLIP, BioCLIP) e modelos visão-linguagem, incluindo fine-tuning com metadados textuais.

## Tecnologias

Python, TensorFlow/Keras, scikit-learn, pandas, Streamlit e Plotly.
