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

y = y['poisonous']


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


eps_range = [i/10 for i in range(1,10)]
min_samples = range(5, 50, 5)
linkage = ['ward', 'complete', 'average', 'single']


def calculate_inertia(X, labels):
    inertia = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia


## KMEANS
for n_clusters in clusters_range:
    for max_iter in max_iter_range:
        kmeans = KMeans(n_clusters=n_clusters,
                        max_iter=max_iter, random_state=0)
        kmeans.fit(X)
        centro = kmeans.cluster_centers_
    
        linha_dados = [
            'KMEANS',
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
            str(contingency_matrix(y, kmeans.labels_).tolist())
        ]
        with open(logs_uri, 'a+') as log_file:
            writer = csv.writer(log_file)
            print(linha_dados)
            
            writer.writerow(linha_dados)



### DBSCAM
for eps in eps_range:
    for min_s in min_samples:
        dbscam = DBSCAN(eps=eps,min_samples=min_s)
        dbscam.fit(X)

        linha_dados = [
            'DBSCAN',
            eps,  # Número de clusters
            min_s,  # Número máximo de iterações
            # Centros dos clusters (convertido para string para salvar no CSV)
            [], ##str(dbscam.cluster_centers_.tolist()),
            float('-inf'),# dbscam.inertia_,  # Soma dos quadrados das distâncias até o centróide mais próximo
            float('-inf'),#  math.sqrt(dbscam.inertia_) / dbscam.n_clusters,  # Coesão
            float('-inf'),  # Coeficiente de Silhueta
            metrics.rand_score(y, dbscam.labels_),  # Rand Score
            metrics.homogeneity_score(y, dbscam.labels_),  # Homogeneidade
            metrics.completeness_score(y, dbscam.labels_),  # Completude
            entropy(dbscam.labels_),  # Entropia
            str(contingency_matrix(y, dbscam.labels_).tolist())
        ]
        with open(logs_uri, 'a+') as log_file:
            writer = csv.writer(log_file)
            print(linha_dados)
            writer.writerow(linha_dados)

### AGNES
for n_clusters in clusters_range:
    for linker in linkage:
        agnes = AgglomerativeClustering(n_clusters=n_clusters,
                        linkage=linker)
        agnes.fit(X)

        linha_dados = [
            'AGNES',
            agnes.n_clusters,  # Número de clusters
            linker,  # Número máximo de iterações
            # Centros dos clusters (convertido para string para salvar no CSV)
            [], ##str(agnes.cluster_centers_.tolist()),
            float('-inf'),# agnes.inertia_,  # Soma dos quadrados das distâncias até o centróide mais próximo
            float('-inf'),#  math.sqrt(agnes.inertia_) / agnes.n_clusters,  # Coesão
            metrics.silhouette_score(X, agnes.labels_),  # Coeficiente de Silhueta
            metrics.rand_score(y, agnes.labels_),  # Rand Score
            metrics.homogeneity_score(y, agnes.labels_),  # Homogeneidade
            metrics.completeness_score(y, agnes.labels_),  # Completude
            entropy(agnes.labels_),  # Entropia
            str(contingency_matrix(y, agnes.labels_).tolist())
        ]
        with open(logs_uri, 'a+') as log_file:
            writer = csv.writer(log_file)
            print(linha_dados)
            writer.writerow(linha_dados)
