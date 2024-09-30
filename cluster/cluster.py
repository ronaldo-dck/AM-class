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
X = X.drop(columns='veil-type')
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

logs_uri = "logs2.csv"


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


eps_range = [i*10 for i in range(1,10)]
print(eps_range)
min_samples = range(50, 501, 50)
linkage = ['ward', 'complete', 'average', 'single']


def calculate_inertia(data, labels):
    """
    Calcula a inércia (soma de quadrados intra-grupo) para um conjunto de dados.

    Parameters:
    data (numpy.ndarray): Matriz de dados (n_samples x n_features).
    labels (numpy.ndarray): Rótulos dos grupos (1D array com o mesmo número de amostras que data).

    Returns:
    float: A inércia total do agrupamento.
    dict: Inércia de cada grupo.
    """
    inertia = 0
    inertia_per_group = {}
    
    # Obter os rótulos únicos
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Selecionar os pontos pertencentes ao grupo atual
        group_points = data[labels == label]
        
        # Calcular o centróide do grupo
        centroid = np.mean(group_points, axis=0)
        
        # Calcular a inércia para o grupo
        group_inertia = np.sum(np.linalg.norm(group_points - centroid, axis=1) ** 2)
        
        # Adicionar à inércia total
        inertia += group_inertia
        
        # Armazenar a inércia do grupo
        inertia_per_group[label] = group_inertia
    return inertia

def calculate_cohesion(X, labels):
    """
    Calcula uma coesão alternativa baseada nas distâncias dentro dos clusters.
    """
    cohesion = 0
    for label in np.unique(labels):
        if label == -1:  # Ignorar ruído do DBSCAN (rótulo -1)
            continue
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        cohesion += np.sum(np.linalg.norm(cluster_points - centroid, axis=1))
    return cohesion / len(np.unique(labels))


def calculate_metrics(method_name, labels, X, y, n_clusters=None, centroids=None, var1=None, var2=None):
    """
    Função central para calcular todas as métricas, adaptando para métodos onde inércia e coesão não existem.
    """
    # Calcula inércia (ou valor análogo) e coesão
    inertia = calculate_inertia(X, labels)
    cohesion = calculate_cohesion(X, labels)

    # Calcula Silhouette Score (somente se houver mais de um cluster)
    silhouette = None
    if len(np.unique(labels)) > 1:
        silhouette = metrics.silhouette_score(X, labels)
        print(silhouette)
    
    # Coleta as demais métricas
    rand_score = metrics.rand_score(y, labels)
    homogeneity = metrics.homogeneity_score(y, labels)
    completeness = metrics.completeness_score(y, labels)

    # Cálculo da entropia
    if len(np.unique(labels)) > 1:  # Verifica se há mais de um cluster
        # Calcula a frequência de rótulos
        value_counts = np.bincount(labels[labels != -1])  # Ignora rótulos de ruído
        probabilities = value_counts / np.sum(value_counts)
        ent = entropy(probabilities)
    else:
        ent = 0  # Ou outro valor que faça sentido no contexto

    cont_matrix = str(contingency_matrix(y, labels).tolist())
    
    linha_dados = [
        method_name,
        n_clusters if n_clusters else len(np.unique(labels)),
        var1,
        var2,
        inertia if inertia else '-',
        cohesion if cohesion else '-',
        silhouette if silhouette else '-',
        rand_score,
        homogeneity,
        completeness,
        ent,
        cont_matrix
    ]
    
    with open(logs_uri, 'a+') as log_file:
        writer = csv.writer(log_file)
        print(linha_dados)
        writer.writerow(linha_dados)


# # KMEANS
# for n_clusters in clusters_range:
#     for max_iter in max_iter_range:
#         kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
#         kmeans.fit(X)
        
#         calculate_metrics(
#             method_name='KMEANS',
#             labels=kmeans.labels_,
#             X=X,
#             y=y,
#             n_clusters=kmeans.n_clusters,
#             centroids=str(kmeans.cluster_centers_.tolist()),
#             var1=n_clusters,
#             var2=max_iter
#         )

# DBSCAN
for eps in eps_range:
    for min_s in min_samples:
        dbscan = DBSCAN(eps=eps, min_samples=min_s)
        dbscan.fit(X)
        
        calculate_metrics(
            method_name='DBSCAN',
            labels=dbscan.labels_,
            X=X,
            y=y,
            var1=eps,
            var2=min_s
        )

# AGNES
for n_clusters in clusters_range:
    for linker in linkage:
        agnes = AgglomerativeClustering(n_clusters=n_clusters, linkage=linker)
        agnes.fit(X)
        
        calculate_metrics(
            method_name='AGNES',
            labels=agnes.labels_,
            X=X,
            y=y,
            n_clusters=agnes.n_clusters,
            var1=n_clusters,
            var2=linker
        )
