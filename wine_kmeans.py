## Trabalho de Análise de Dados e Clusterização - Dataset Wine
## Técnicas: K-means (k=3) e análise de correlação

## 0. Importação das bibliotecas
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

## Pasta onde os gráficos serão salvos
os.makedirs("images", exist_ok=True)

## 1. Importação e visualização dos dados
wine = load_wine()

## Carrega os dados em um DataFrame do pandas
df = pd.DataFrame(wine.data, columns=wine.feature_names)

## Primeiros registros e informações gerais
print("Primeiros registros:")
print(df.head())
print("\nDimensões:", df.shape)
print("\nValores nulos por coluna:")
print(df.isnull().sum().sum(), "valores nulos no total")
print("\nEstatísticas descritivas:")
print(df.describe().T.round(2))

## 2. Aplicação do K-means (3 clusters)
## As variáveis estão em escalas muito diferentes (ex.: proline ~ 700,
## hue ~ 1). Como o K-means usa distância euclidiana, padronizamos os
## dados (média 0, desvio 1) para que nenhuma variável domine o cálculo.
X = df.copy()
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)  ## coluna com o cluster de cada vinho

print("\nQuantidade de vinhos por cluster:")
print(df["cluster"].value_counts().sort_index())

## (Extra) Comparação com as classes reais de uva, apenas para validação
print("\nComparação cluster x classe real:")
print(pd.crosstab(df["cluster"], wine.target, rownames=["cluster"], colnames=["classe real"]))
print("Adjusted Rand Index:", round(adjusted_rand_score(wine.target, df["cluster"]), 3))

## 3. Análise de correlação
## Matriz de correlação (somente variáveis originais, sem a coluna cluster)
corr = df.drop(columns="cluster").corr()

## Visualização gráfica (heatmap)
plt.figure(figsize=(13, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, linewidths=0.5)
plt.title("Matriz de correlação - Dataset Wine")
plt.tight_layout()
plt.savefig("images/matriz_correlacao.png", dpi=150)
plt.show()

## Pares com correlação moderada (0,3 <= |r| <= 0,7)
pares = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):  ## só triângulo superior (sem repetir)
        r = corr.iloc[i, j]
        if 0.3 <= abs(r) <= 0.7:
            pares.append((cols[i], cols[j], round(r, 3)))

df_pares = (pd.DataFrame(pares, columns=["variavel_1", "variavel_2", "correlacao"])
              .sort_values("correlacao", key=abs, ascending=False)
              .reset_index(drop=True))
print(f"\nPares com correlação moderada (0,3 a 0,7): {len(df_pares)}")
print(df_pares.to_string())

## 4. Visualização dos clusters
## Variáveis escolhidas a partir da análise de correlação
var_x = "flavanoids"
var_y = "proline"
print(f"\nCorrelação entre {var_x} e {var_y}: {corr.loc[var_x, var_y]:.3f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x=var_x, y=var_y, hue="cluster",
                palette="Set1", s=60, edgecolor="black")

## Centróides (convertidos de volta para a escala original)
centros = pd.DataFrame(
    StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_),
    columns=X.columns,
)
plt.scatter(centros[var_x], centros[var_y], c="black", marker="X", s=250,
            label="Centróides")
plt.title(f"Clusters K-means: {var_x} x {var_y}")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("images/clusters_scatter.png", dpi=150)
plt.show()

## Médias das variáveis escolhidas por cluster (apoia a interpretação)
print("\nMédia por cluster:")
print(df.groupby("cluster")[[var_x, var_y]].mean().round(2))

## Conclusão
## O K-means com 3 grupos conseguiu separar bem os vinhos, e os grupos ficaram
## quase iguais aos três tipos de uva do dataset. Na correlação, vi que muitas
## variáveis têm relação média entre si, então cada uma ajuda com uma informação
## diferente. Escolhi flavonoides e prolina, e no gráfico deu para ver os três
## grupos separados: um com poucos flavonoides, um com muita prolina e muitos
## flavonoides, e um no meio. Isso mostra que escolher variáveis pela correlação
## ajuda a visualizar os grupos, mas só duas variáveis não mostram tudo.
