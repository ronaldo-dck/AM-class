import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata
from utils import gerar_arquiteturas




# dados = pd.read_csv("./datasets/diabetes_012_health_indicators_BRFSS2015.csv")
dados = pd.read_csv("./base_de_dados_normalizada.csv")


log_filename = 'logs.csv'


log_knn = 'log_params_knn.csv'
log_ad = 'log_params_ad.csv'
log_mlp = 'log_params_mlp.csv'
log_svm = 'log_params_svm.csv'



t_init = time.time()
for interation in tqdm(range(20), desc='Processing', unit='test'):
    dados = shuffle(dados, random_state=interation)

    X = np.array(dados.iloc[:,:-1])
    Y = np.array(dados.iloc[:,-1]).ravel()
    
    # x_treino, x_temp, y_treino, y_temp = train_test_split(
    #     X, Y, test_size=0.5, stratify=Y, random_state=interation)
    # x_validacao, x_teste, y_validacao, y_teste = train_test_split(
    #     x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=interation)
    X_reduzido, _, Y_reduzido, _ = train_test_split(
    X, Y, test_size=0.90, stratify=Y, random_state=1)

    x_treino, x_temp, y_treino, y_temp = train_test_split(
        X_reduzido, Y_reduzido, test_size=0.5, stratify=Y_reduzido, random_state=1)

    x_validacao, x_teste, y_validacao, y_teste = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=1)

    

##########################################################################################################

    maior = -1
    for j in ("distance", "uniform"):
        for i in range(1, 51):
            KNN = KNeighborsClassifier(n_neighbors=i, weights=j)
            KNN.fit(x_treino, y_treino)
            opiniao = KNN.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            print("K: ", i, " Métrica: ", j, " Acc: ", Acc)
            if (Acc > maior):
                maior = Acc
                Melhor_k = i
                Melhor_metrica = j


    KNN = KNeighborsClassifier(n_neighbors=Melhor_k,weights=Melhor_metrica)
    KNN.fit(x_treino,y_treino)
    opiniao_KNN  = KNN.predict(x_teste)
    prob_KNN     = KNN.predict_proba(x_teste)
    
    accuracy_KNN = accuracy_score(y_teste, opiniao)
    with open(log_knn, 'a+') as log_file:
        log_file.write(f'{interation},{Melhor_k},{Melhor_metrica}')

###########################################################################################################

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

                        if (Acc > maior):
                            maior = Acc
                            crit = j
                            md = i
                            msl = k
                            mss = l
                            split = m


    AD = DecisionTreeClassifier(criterion=crit,max_depth=md,min_samples_leaf=msl,min_samples_split=mss,splitter=split)
    AD.fit(x_treino,y_treino)
    opiniao_AD = AD.predict(x_teste)
    prob_AD = AD.predict_proba(x_teste)
    accuracy_AD = accuracy_score(y_teste, opiniao)


    with open(log_ad, 'a+') as log_file:
        log_file.write(f'{interation},{crit},{md},{msl},{mss},{split}')

    
    ################################################
    
    NB = GaussianNB()
    NB.fit(x_treino,y_treino)
    opiniao_NB = NB.predict(x_teste)
    prob_NB = NB.predict_proba(x_teste)
    accuracy_NB = accuracy_score(y_teste, opiniao_NB)

#     ################################################


    best_mlp_acc = -1
    best_i_mlp = 0
    best_j_mlp = 0
    best_k_mlp = 0
    best_func_mlp = 0
    best_y_pred = 0


    arquiteturas = gerar_arquiteturas()

    for arq in arquiteturas:
        for learning_rate in ('constant', 'invscaling', 'adaptive'):
            for epocas in (50, 100, 150, 200):
                for l in ('identity', 'logistic', 'tanh', 'relu'):

                    MLP = MLPClassifier(
                        hidden_layer_sizes=arq, verbose=True, learning_rate=learning_rate, max_iter=epocas, activation=l, early_stopping=True)
                    
                    MLP.fit(x_treino, y_treino)

                    opiniao = MLP.predict(x_validacao)
                    accuracy_mlp = accuracy_score(y_validacao, opiniao)

                    if (accuracy_mlp > best_mlp_acc):
                            best_mlp_acc = accuracy_mlp
                            best_i_mlp = arq
                            best_learning_rate_mlp = learning_rate
                            best_epocas_mlp = epocas
                            best_func_mlp = l

            


    MLP = MLPClassifier(hidden_layer_sizes=best_i_mlp, learning_rate=best_learning_rate_mlp, max_iter=best_epocas_mlp, activation=best_func_mlp)
    MLP.fit(x_treino,y_treino)
    opiniao_MLP = MLP.predict(x_teste)
    prob_MLP = MLP.predict_proba(x_teste)
    accuracy_MLP = accuracy_score(y_teste, opiniao_MLP)
    prob_opina = MLP.predict_proba(x_teste)

    with open(log_mlp, 'a+') as log_file:
        log_file.write(f'{interation},{best_i_mlp},{best_learning_rate_mlp},{best_epocas_mlp},{best_func_mlp}')


# ##########################################################################################################
    ## SVM


    maior = -1

    # Grid search nos parâmetros kernel e C
    for kernel in ("linear", "poly", "rbf", "sigmoid"):
        for C in [0.1, 1, 10, 100, 200]:
            svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
            svm.fit(x_treino, y_treino)
            opiniao = svm.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            print(f"Kernel: {kernel}, C: {C}, Acc: {Acc:.2f}")
            if Acc > maior:
                maior = Acc
                best_kernel = kernel
                best_C = C

    svm = SVC(kernel=best_kernel, C=best_C, probability=True, random_state=42)
    svm.fit(x_treino, y_treino)
    opiniao_SVM = svm.predict(x_teste)
    prob_SVM = svm.predict_proba(x_teste)
    accuracy_SVM = accuracy_score(y_teste, opiniao_SVM)
    with open(log_svm, 'a+') as log_file:
        log_file.write(f'{1},{best_kernel},{best_C}')


##############################################################################

    #soma 
    soma_prob = prob_KNN + prob_AD + prob_NB + prob_MLP + prob_SVM
    opiniao_soma = np.argmax(soma_prob, axis=1)
    accuracy_soma = accuracy_score(y_teste, opiniao_soma)
    print(f"Accuracy Regra da Soma: {accuracy_soma}")

    #voto
    opinioes = np.array([opiniao_KNN, opiniao_AD, opiniao_NB, opiniao_MLP, opiniao_SVM], dtype=int)
    opiniao_voto_majoritario = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=opinioes)
    accuracy_voto_majoritario = accuracy_score(y_teste, opiniao_voto_majoritario)
    print(f"Accuracy Voto Majoritário: {accuracy_voto_majoritario}")

    # Borda Count - GPT ---- Me parece certo
    ranks = np.array([rankdata(opiniao_KNN, method='min'),
                      rankdata(opiniao_AD, method='min'),
                      rankdata(opiniao_NB, method='min'),
                      rankdata(opiniao_MLP, method='min'),
                      rankdata(opiniao_SVM, method='min')])
    soma_ranks = np.sum(ranks, axis=0)
    opiniao_borda_count = np.argmax(soma_ranks, axis=0)
    accuracy_borda_count = accuracy_score(y_teste, opiniao_borda_count)

    # Log dos resultados
    with open(log_filename, 'a+') as log_file:
        log_file.write(f"{interation}, KNN, {accuracy_KNN}\n")
        log_file.write(f"{interation}, NB, {accuracy_NB}\n")
        log_file.write(f"{interation}, MLP, {accuracy_MLP}\n")
        log_file.write(f"{interation}, AD, {accuracy_AD}\n")
        log_file.write(f"{interation}, SVM, {accuracy_SVM}\n")
        log_file.write(f"{interation}, Soma, {accuracy_soma}\n")
        log_file.write(f"{interation}, Voto Majoritário, {accuracy_voto_majoritario}\n")
        log_file.write(f"{interation}, Borda Count, {accuracy_borda_count}\n")



print("TEMPO TOTAL: ", time.time() - t_init)