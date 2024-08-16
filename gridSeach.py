import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier




dados = pd.read_csv("./datasets/diabetes_012_health_indicators_BRFSS2015.csv")

log_filename = 'logs.csv'




for interation in tqdm(range(20), desc='Processing', unit='row'):
    dados = shuffle(dados, random_state=interation)

    X = dados.iloc[:, 1:]
    Y = dados.iloc[:, 0:1]

    print(Y)

    x_treino, x_temp, y_treino, y_temp = train_test_split(
        X, Y, test_size=0.5, stratify=Y, random_state=interation)
    x_validacao, x_teste, y_validacao, y_teste = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=interation)

    t_init = time.time()

# ##########################################################################################################

    maior = -1
    for j in ("distance", "uniform"):
        for i in range(1, 50):
            KNN = KNeighborsClassifier(n_neighbors=i, weights=j)
            KNN.fit(x_treino, y_treino)
            opiniao = KNN.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            print("K: ", i, " Métrica: ", j, " Acc: ", Acc)
            if (Acc > maior):
                maior = Acc
                Melhor_k = i
                Melhor_metrica = j


    KNN = KNeighborsClassifier(n_neighbors=i, weights=j)
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    accuracy_KNN = accuracy_score(y_teste, opiniao)


# ##########################################################################################################

    maior = -1
    for j in ("entropy", "gini"):  # criterion
        for i in range(1, 11):  # max_depth
            for k in range(1, 11):  # min_samples_leaf
                for l in range(2, 16):  # min_samples_split
                    for m in ('best', 'random'):  # splitter
                        AD = DecisionTreeClassifier(
                            criterion=j, max_depth=i, min_samples_leaf=k, min_samples_split=l, splitter=m)
                        AD.fit(x_treino, y_treino)
                        opiniao = AD.predict(x_validacao)
                        Acc = accuracy_score(y_validacao, opiniao)
                        print("Criterion: ", j, " max_depth: ", i, " min_samples_leaf: ",
                              k, " min_samples_split: ", l, " splitter: ", m, " Acc: ", Acc)
                        if (Acc > maior):
                            maior = Acc
                            crit = j
                            md = i
                            msl = k
                            mss = l
                            split = m

    AD = DecisionTreeClassifier(criterion=crit, max_depth=md,
                                min_samples_leaf=msl, min_samples_split=mss, splitter=split)
    AD.fit(x_treino, y_treino)
    opiniao = AD.predict(x_teste)
    accuracy_AD = accuracy_score(y_teste, opiniao)

#     ################################################

    NB = GaussianNB()
    NB.fit(x_treino, y_treino)
    opiniao = NB.predict(x_teste)
    accuracy_NB = accuracy_score(y_teste, opiniao)

#     ################################################


    best_mlp_acc = -1
    best_i_mlp = 0
    best_j_mlp = 0
    best_k_mlp = 0
    best_l_mlp = 0
    best_y_pred = 0

    from utils import gerar_arquiteturas

    arquiteturas = gerar_arquiteturas()

    for arq in arquiteturas:
        for learning_rate in ('constant', 'invscaling', 'adaptive'):
            for epocas in (50, 100, 150, 300, 500):
                for l in ('identity', 'logistic', 'tanh', 'relu'):

                    MLP = MLPClassifier(
                        hidden_layer_sizes=arq, verbose=True, learning_rate=learning_rate, max_iter=epocas, activation=l, early_stopping=True)
                    MLP.fit(x_treino, y_treino)

                    opiniao = MLP.predict(x_validacao)
                    accuracy_mlp = accuracy_score(y_validacao, opiniao)


    inicio = time.time()
    MLP = MLPClassifier(hidden_layer_sizes=(42,42), verbose=True, max_iter=10, activation='relu')
    print("PARAMETROS", MLP.get_params())
    # MLP = MLPClassifier(hidden_layer_sizes=(best_i_mlp, best_i_mlp, 1),
    #                     learning_rate=best_j_mlp, max_iter=best_k_mlp, activation=best_l_mlp)
    MLP.fit(x_treino, y_treino)
    print("TEMPO: ", time.time() - inicio)

    opiniao = MLP.predict(x_teste)
    
    accuracy_MLP = accuracy_score(y_teste, opiniao)

    prob_opina = MLP.predict_proba(x_teste)
    print
# ##########################################################################################################
    ## SVM


    maior = -1

    # Grid search nos parâmetros kernel e C
    for kernel in ("linear", "poly", "rbf", "sigmoid"):
        for C in [0.1, 1, 10, 100, 1000]:
            svm = SVC(kernel=kernel, C=C, random_state=42)
            svm.fit(x_treino, y_treino)
            opiniao = svm.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            print(f"Kernel: {kernel}, C: {C}, Acc: {Acc:.2f}")
            if Acc > maior:
                maior = Acc
                best_kernel = kernel
                best_C = C

    print("\nMelhor configuração para o SVM")
    print(f"Kernel: {best_kernel}, C: {best_C}, Acc: {maior:.2f}")

    print("\n\nDesempenho sobre o conjunto de teste")
    svm = SVC(kernel=best_kernel, C=best_C, random_state=42)
    svm.fit(x_treino, y_treino)
    opiniao = svm.predict(x_teste)
    accuracy_SVM = accuracy_score(y_teste, opiniao)
    print(f"Accuracy no conjunto de teste: {accuracy_SVM:.2f}")
    SVM = SVC()

    SVM.fit(x_treino,y_treino)
    opiniao = SVM.predict(x_teste)
    accuracy_SVM = accuracy_score(y_teste, opiniao)



    with open(log_filename, 'a+') as log_file:
        log_file.write(interation, 'KNN', accuracy_KNN)
        log_file.write(interation, 'NB', accuracy_NB)
        log_file.write(interation, 'MLP', accuracy_MLP)
        log_file.write(interation, 'AD', accuracy_AD)
        log_file.write(interation, 'SVM', accuracy_SVM)
    pass


