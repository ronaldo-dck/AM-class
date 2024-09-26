import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import entropy
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
import csv

from ucimlrepo import fetch_ucirepo

# fetch dataset
mushroom = fetch_ucirepo(id=73)


# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets


X = pd.DataFrame(X)
y = pd.DataFrame(y)


for c in X.columns:
    classes = X[c].unique()
    nums_cl = [i for i in range(len(classes))]
    # Cria um dicionário de mapeamento das classes para os números
    mapping = dict(zip(classes, nums_cl))

    # Substitui as classes pelas suas representações numéricas
    X[c] = X[c].map(mapping)


mapeamento = {
    'e': 0,
    'p': 1
}
y['poisonous'] = y['poisonous'].replace(mapeamento)

#############################################################################################

logs_uri = "logs.csv"


"""
    KMEANS:
        clusters: 2 - 6
        max: 100 - 1000 - 100

    DBSAM:
        eps: 0.1 - 0.9 - 0.1
        min-samples: 5 - 50 - 5
    AGNES:
        clustes: 2 - 6
        link : all
"""

#############################################################################################


# kmeans = KMeans(n_clusters=2, max_iter=3000,random_state=0)
# kmeans.fit(X)
# centro = kmeans.cluster_centers_

clusters_range = range(2, 7)
max_iter_range = range(100, 1100, 100)

for n_clusters in clusters_range:
    for max_iter in max_iter_range:
        kmeans = KMeans(n_clusters=n_clusters,
                        max_iter=max_iter, random_state=0)
        kmeans.fit(X)
        centro = kmeans.cluster_centers_
    
        linha_dados = [
            kmeans.n_clusters,  # Número de clusters
            kmeans.max_iter,  # Número máximo de iterações
            # Centros dos clusters (convertido para string para salvar no CSV)
            str(kmeans.cluster_centers_.tolist()),
            kmeans.inertia_,  # Soma dos quadrados das distâncias até o centróide mais próximo
            math.sqrt(kmeans.inertia_) / kmeans.n_clusters,  # Coesão
            metrics.silhouette_score(X, kmeans.labels_),  # Coeficiente de Silhueta
            metrics.rand_score(y, kmeans.labels_),  # Rand Score
            metrics.homogeneity_score(y, kmeans.labels_),  # Homogeneidade
            metrics.completeness_score(y, kmeans.labels_),  # Completude
            entropy(kmeans.labels_),  # Entropia
            # Matriz de contingência (convertido para string)
            str(contingency_matrix(y, kmeans.labels_).tolist())
        ]

    with open(logs_uri, 'a+') as log_file:
        writer = csv.writer(log_file)
        print(writer)
        writer.writerow(linha_dados)


y = y['poisonous']
print(y)
print(kmeans.labels_)
