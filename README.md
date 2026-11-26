# Clusterização de Vinhos com K-means

Análise exploratória e agrupamento (clusterização) do dataset Wine usando K-means e análise de correlação, em Python.

Trabalho acadêmico desenvolvido no curso de Ciência de Dados.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

## Sobre o projeto

Este trabalho foi realizado a partir do **scikit-learn**, biblioteca de aprendizado de máquina em Python. Ele é usado em todas as etapas principais:

| Etapa | Recurso do scikit-learn |
|---|---|
| Carregamento dos dados | `sklearn.datasets.load_wine` |
| Padronização das variáveis | `sklearn.preprocessing.StandardScaler` |
| Agrupamento | `sklearn.cluster.KMeans` |
| Validação dos clusters | `sklearn.metrics.adjusted_rand_score` |

O objetivo é verificar se o K-means, sem conhecer o tipo de uva, consegue separar os vinhos em grupos coerentes, e usar a correlação entre as variáveis para escolher quais delas visualizar.

## Dataset

- Fonte: dataset Wine (UCI), disponível no scikit-learn.
- 178 vinhos e 13 variáveis químicas (álcool, flavonoides, prolina, intensidade de cor, entre outras).
- Sem valores nulos.
- Os vinhos vêm de 3 tipos de uva. Essa informação não é usada no agrupamento, apenas para conferir o resultado no final.

## Metodologia

1. **Exploração dos dados:** primeiros registros, dimensões, valores nulos e estatísticas descritivas.
2. **Padronização:** as variáveis têm escalas muito diferentes (a prolina passa de 700, o matiz fica perto de 1). Como o K-means usa distância euclidiana, foi aplicado o `StandardScaler` (média 0, desvio 1) para que nenhuma variável domine o cálculo.
3. **K-means com 3 clusters** (`random_state=42`, `n_init=10`).
4. **Validação:** comparação dos clusters com as classes reais de uva, usando tabela cruzada e o Adjusted Rand Index.
5. **Correlação:** matriz de correlação e lista dos pares com correlação moderada (0,3 <= |r| <= 0,7).
6. **Visualização:** gráfico de dispersão dos clusters com variáveis escolhidas a partir da correlação, incluindo os centróides.

## Resultados

### Tamanho dos clusters e comparação com as uvas reais

| Cluster | Vinhos | Classe 0 | Classe 1 | Classe 2 |
|:-------:|:------:|:--------:|:--------:|:--------:|
| 0 | 65 | 0 | 65 | 0 |
| 1 | 51 | 0 | 3 | 48 |
| 2 | 62 | 59 | 3 | 0 |

- Adjusted Rand Index: 0,897 (1,0 seria uma correspondência perfeita).
- Apenas 6 dos 178 vinhos ficaram em um grupo diferente da sua uva de origem.

### Matriz de correlação

Muitas variáveis têm correlação moderada entre si (32 pares entre 0,3 e 0,7 em valor absoluto), então cada uma traz uma informação um pouco diferente. As correlações mais fortes são entre fenóis totais e flavonoides (0,86) e entre flavonoides e OD280/OD315 (0,79).

![Matriz de correlação](images/matriz_correlacao.png)

### Clusters: flavonoides x prolina

Foram escolhidas as variáveis flavonoides e prolina (correlação de 0,49 entre elas) para visualizar os grupos.

![Clusters K-means](images/clusters_scatter.png)

| Cluster | Flavonoides (média) | Prolina (média) |
|:-------:|:-------------------:|:---------------:|
| 0 | 2,05 | 510,17 |
| 1 | 0,82 | 619,06 |
| 2 | 3,00 | 1100,23 |

## Conclusões

- O K-means com 3 grupos separou bem os vinhos, e os grupos ficaram quase iguais aos três tipos de uva do dataset.
- No gráfico é possível ver os três grupos: um com poucos flavonoides (cluster 1), um com muita prolina e muitos flavonoides (cluster 2) e um intermediário (cluster 0).
- Escolher as variáveis pela correlação ajuda a visualizar os grupos, mas apenas duas variáveis não mostram tudo: há vinhos misturados na região de fronteira entre os grupos.

## Como executar

```bash
# Clonar o repositório
git clone https://github.com/lav-vv/wine-kmeans-clusterizacao.git
cd wine-kmeans-clusterizacao

# (Opcional) criar um ambiente virtual
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Instalar as dependências
pip install -r requirements.txt

# Executar
python wine_kmeans.py
```

O script imprime os resultados no terminal e salva os gráficos na pasta `images/`.

## Estrutura do projeto

```
wine-kmeans-clusterizacao/
├── images/
│   ├── clusters_scatter.png
│   └── matriz_correlacao.png
├── .gitignore
├── README.md
├── requirements.txt
└── wine_kmeans.py
```

## Autora

Lavinia de Jesus, estudante de Ciência de Dados.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lav%C3%ADnia-de-jesus-pe%C3%A7anha-018245283) [![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=flat&logo=instagram&logoColor=white)](https://www.instagram.com/lavj__/)
