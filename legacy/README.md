# Código de 2021 (histórico)

Arquivos mantidos como registro da primeira versão do projeto. Não são mais usados.

- `Crawler_Museu_Filo.py` e `Crawler_Museu_Filo_e_Classe.py`: crawlers do site do MNHN, escritos para o Google Colab. Dependem do layout do site na época e provavelmente não funcionam mais.
- `main_antigo.py`: app Streamlit de análise exploratória.
- `LoadModel_Streamlit_antigo.py`: app Streamlit do classificador.

## Problemas conhecidos

- **Foto herdada**: em `coleta_info`, quando a página não tinha imagem, `find_all` retornava lista vazia (nunca `None`), o laço não executava e `foto` mantinha o valor do espécime anterior. Isso gerou 2.981 registros com a foto errada no CSV. A correção está em `fossil/metadata.py`.
- **Várias imagens por espécime**: quando a página tinha mais de uma imagem, todas eram salvas com o mesmo nome de arquivo, e só a última ficava.
- **Caminhos fixos do Google Drive** e uso de `os.chdir`, o que impede rodar fora do Colab.
- `main_antigo.py` usa `st.beta_columns` (removido do Streamlit) e a segunda linha de imagens reaproveitava as colunas da primeira. A seção de Bryozoa mostrava código e título de Foraminifera.
- `LoadModel_Streamlit_antigo.py` carrega o modelo de um caminho do Drive e chama de "precisão" a probabilidade do softmax.
