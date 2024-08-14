import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from scipy.stats import rankdata



dados = pd.read_csv("./datasets/diabetes_012_health_indicators_BRFSS2015.csv")

log_filename = 'logs.csv'

for interation in tqdm(range(20), desc='Processing', unit='row'):
    dados = shuffle(dados, random_state=interation)

    X = dados.iloc[:,1:]
    Y = dados.iloc[:,0:1]

    x_treino,x_temp,y_treino,y_temp=train_test_split(X,Y,test_size=0.5,stratify=Y, random_state=interation)
    x_validacao,x_teste,y_validacao,y_teste=train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp, random_state=interation)

    t_init = time.time()

    maior = -1
    for j in ("distance","uniform"):
        for i in range(1,50): 
            KNN = KNeighborsClassifier(n_neighbors=i,weights=j)
            KNN.fit(x_treino,y_treino)
            opiniao = KNN.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            print("K: ",i," Métrica: ",j," Acc: ",Acc)
            if (Acc > maior):
                maior = Acc
                Melhor_k = i
                Melhor_metrica = j

    print("\nMelhor configuração para o KNN")
    print("K: ", Melhor_k," Métrica: ", Melhor_metrica," Acurácia sobre a validação: ",maior)


    print("\n\nDesempenho sobre o conjunto de teste")
    KNN = KNeighborsClassifier(n_neighbors=i,weights=j)
    KNN.fit(x_treino,y_treino)
    opiniao_KNN = KNN.predict(x_teste)
    prob_KNN = KNN.predict_proba(x_teste)
    accuracy_KNN = accuracy_score(y_teste, opiniao)

    print("TEMPO:", time.time()-t_init)

##########################################################################################################

    maior = -1
    for j in ("entropy","gini"):  #criterion
        for i in range (1,11):      #max_depth
            for k in range (1,11):    #min_samples_leaf
                for l in range (2,16):  #min_samples_split
                    for m in ('best','random'): #splitter
                        AD = DecisionTreeClassifier(criterion=j,max_depth=i,min_samples_leaf=k,min_samples_split=l,splitter=m)
                        AD.fit(x_treino,y_treino)
                        opiniao = AD.predict(x_validacao)
                        Acc = accuracy_score(y_validacao, opiniao)
                        print("Criterion: ",j," max_depth: ",i," min_samples_leaf: ",k," min_samples_split: ",l," splitter: ",m," Acc: ",Acc)
                        if (Acc > maior):
                            maior = Acc
                            crit = j
                            md = i
                            msl = k
                            mss = l
                            split = m

    print("\nMelhor configuração para a AD")
    print("Criterion: ",crit," max_depth: ",md," min_samples_leaf: ",msl," min_samples_split: ",mss," splitter: ",split," Acc: ",maior)

    print("\n\nDesempenho sobre o conjunto de teste")
    AD = DecisionTreeClassifier(criterion=crit,max_depth=md,min_samples_leaf=msl,min_samples_split=mss,splitter=split)
    AD.fit(x_treino,y_treino)
    opiniao_AD = AD.predict(x_teste)
    prob_AD = AD.predict_proba(x_teste)
    accuracy_AD = accuracy_score(y_teste, opiniao)
    
    
    ################################################
    
    NB = BernoulliNB()
    NB.fit(x_treino,y_treino)
    opiniao_NB = NB.predict(x_teste)
    prob_NB = NB.predict_proba(x_teste)
    accuracy_NB = accuracy_score(y_teste, opiniao_NB)

    
    ################################################

    # 21
    ### 21 - 42 nodos na camada escondidas
    ### 1 - 2 camadas escondidas
    ##  
    ### 3 PASSO
    # 3

    best_mlp_acc = -1
    best_i_mlp = 0
    best_j_mlp = 0
    best_k_mlp = 0
    best_l_mlp = 0
    best_y_pred = 0

    for camadas in range(1,3,1):
        for nodos in (21, 43, 3):
            for nodos2 in (21, 43, 3):
                for learning_rate in ('constant','invscaling', 'adaptive'):
                    for epocas in (50,100,150,300,500):
                        for l in ('identity', 'logistic', 'tanh', 'relu'):
                            if camadas == 1:
                                discritor_net = (nodos,)
                            else:
                                discritor_net = (nodos, nodos2,)
                            MLP = MLPClassifier(hidden_layer_sizes=discritor_net, learning_rate=learning_rate, max_iter=epocas, activation=l )
                            MLP.fit(x_treino,y_treino)

                            y_pred_mlp = MLP.predict(x_validacao)
                            accuracy_mlp = accuracy_score(y_validacao, opiniao)

                            if (accuracy_mlp > best_mlp_acc):
                                best_mlp_acc = accuracy_mlp
                                best_i_mlp = i
                                best_learning_rate_mlp = learning_rate
                                best_epocas_mlp = epocas
                                best_l_mlp = l
                                best_y_pred = y_pred_mlp

            

    print("Acc do  MLP:",best_mlp_acc)
    print("C =", best_i_mlp)
    print("Kernel =", best_learning_rate_mlp)
    print("Max iter =", best_epocas_mlp)
    print("Activation =", best_l_mlp)
    
    # for hidden_layers in range(20):
    MLP = MLPClassifier(hidden_layer_sizes=(best_i_mlp,best_i_mlp,1), learning_rate=best_j_mlp, max_iter=best_k_mlp, activation=best_l_mlp)
    MLP.fit(x_treino,y_treino)
    opiniao_MLP = MLP.predict(x_teste)
    prob_MLP = MLP.predict_proba(x_teste)
    accuracy_MLP = accuracy_score(y_teste, opiniao_MLP)

    
    ################################################
    maior = -1

    # Grid search nos parâmetros kernel e C
    for kernel in ("linear", "poly", "rbf", "sigmoid"):
        for C in [0.1, 1, 10, 100, 1000]:
            svm = SVC(kernel=kernel, C=C, random_state=42)
            svm.fit(x_treino,y_treino)
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
    opiniao_SVM = svm.predict(x_teste)
    prob_SVM = svm.predict_proba(x_teste)
    accuracy_SVM = accuracy_score(y_teste, opiniao_SVM)
    print(f"Accuracy no conjunto de teste: {accuracy_SVM:.2f}")
    SVM = SVC()
    
    # SVM.fit(x_treino,y_treino)
    # opiniao = SVM.predict(x_teste)
    # accuracy_SVM = accuracy_score(y_teste, opiniao)


    # with open(log_filename, 'a+') as log_file:

    #         log_file.write(interation, 'KNN', accuracy_KNN)
    #         log_file.write(interation, 'NB', accuracy_NB)
    #         log_file.write(interation, 'MLP', accuracy_MLP)
    #         log_file.write(interation, 'AD', accuracy_AD)
    #         log_file.write(interation, 'SVM', accuracy_SVM)

    # pass

    #soma 
    soma_prob = prob_KNN + prob_AD + prob_NB + prob_MLP + prob_SVM
    opiniao_soma = np.argmax(soma_prob, axis=1)
    accuracy_soma = accuracy_score(y_teste, opiniao_soma)
    print(f"Accuracy Regra da Soma: {accuracy_soma}")

    #voto
    opinioes = np.array([opiniao_KNN, opiniao_AD, opiniao_NB, opiniao_MLP, opiniao_SVM])
    opiniao_voto_majoritario = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=opinioes) # nao sei se isso ta correto
    accuracy_voto_majoritario = accuracy_score(y_teste, opiniao_voto_majoritario)
    print(f"Accuracy Voto Majoritário: {accuracy_voto_majoritario}")

    # Borda Count - GPT
    ranks = np.array([rankdata(opiniao_KNN, method='min'),
                      rankdata(opiniao_AD, method='min'),
                      rankdata(opiniao_NB, method='min'),
                      rankdata(opiniao_MLP, method='min'),
                      rankdata(opiniao_SVM, method='min')])
    soma_ranks = np.sum(ranks, axis=0)
    opiniao_borda_count = np.argmax(soma_ranks, axis=0)
    accuracy_borda_count = accuracy_score(y_teste, opiniao_borda_count)
    print(f"Accuracy Borda Count: {accuracy_borda_count}")

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