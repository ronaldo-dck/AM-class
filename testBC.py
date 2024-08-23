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


log_filename = 'logs2.csv'

log_knn = 'log_params_knn.csv'
log_ad = 'log_params_ad.csv'
log_mlp = 'log_params_mlp.csv'
log_svm = 'log_params_svm.csv'

params_knn = pd.read_csv('log_params_knn.csv')
params_ad  = pd.read_csv('log_params_ad.csv')
params_mlp = pd.read_csv('log_params_mlp.csv', delimiter=';')
params_svm = pd.read_csv('log_params_svm.csv')


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
    X, Y, test_size=0.99, stratify=Y, random_state=1)

    x_treino, x_temp, y_treino, y_temp = train_test_split(
        X_reduzido, Y_reduzido, test_size=0.5, stratify=Y_reduzido, random_state=1)

    x_validacao, x_teste, y_validacao, y_teste = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=1)

    

##########################################################################################################

    Melhor_k = params_knn['k'][interation]
    Melhor_metrica = params_knn['w'][interation]

    KNN = KNeighborsClassifier(n_neighbors=Melhor_k,weights=Melhor_metrica)
    KNN.fit(x_treino,y_treino)
    opiniao_KNN  = KNN.predict(x_teste)
    prob_KNN     = KNN.predict_proba(x_teste)
    
    accuracy_KNN = accuracy_score(y_teste, opiniao_KNN)

###########################################################################################################

    
    crit = params_ad['criterion'][interation]
    md = params_ad['max_depth'][interation]
    msl = params_ad['min_samples_leaf'][interation]
    mss = params_ad['min_samples_split'][interation]
    split = params_ad['splitter'][interation]


    AD = DecisionTreeClassifier(criterion=crit,max_depth=md,min_samples_leaf=msl,min_samples_split=mss,splitter=split)
    AD.fit(x_treino,y_treino)
    opiniao_AD = AD.predict(x_teste)
    prob_AD = AD.predict_proba(x_teste)
    accuracy_AD = accuracy_score(y_teste, opiniao_AD)

    
    ################################################
    
    NB = GaussianNB()
    NB.fit(x_treino,y_treino)
    opiniao_NB = NB.predict(x_teste)
    prob_NB = NB.predict_proba(x_teste)
    accuracy_NB = accuracy_score(y_teste, opiniao_NB)

#     ################################################


    best_i_mlp = params_mlp['arq'][interation].strip('[]')
    string_list = best_i_mlp.split(', ')
    best_i_mlp = list(map(int, string_list))
    best_learning_rate_mlp = params_mlp['learning'][interation]

    best_epocas_mlp = params_mlp['epocas'][interation]
    best_func_mlp = params_mlp['func'][interation]

   
    MLP = MLPClassifier(hidden_layer_sizes=best_i_mlp, learning_rate=best_learning_rate_mlp, max_iter=best_epocas_mlp, activation=best_func_mlp)
    MLP.fit(x_treino,y_treino)
    opiniao_MLP = MLP.predict(x_teste)
    prob_MLP = MLP.predict_proba(x_teste)
    accuracy_MLP = accuracy_score(y_teste, opiniao_MLP)
    prob_opina = MLP.predict_proba(x_teste)


    print("MLP",interation, best_i_mlp)

# ##########################################################################################################
    ## SVM


    best_kernel = params_svm['kernel'][interation]
    best_C = params_svm['C'][interation]


    svm = SVC(kernel=best_kernel, C=best_C, probability=True, random_state=42)
    svm.fit(x_treino, y_treino)
    opiniao_SVM = svm.predict(x_teste)
    prob_SVM = svm.predict_proba(x_teste)
    accuracy_SVM = accuracy_score(y_teste, opiniao_SVM)



##############################################################################

    #soma 
    soma_prob = prob_KNN + prob_AD + prob_NB + prob_MLP + prob_SVM
    opiniao_soma = np.argmax(soma_prob, axis=1)
    accuracy_soma = accuracy_score(y_teste, opiniao_soma)
    print(f"Accuracy Regra da Soma: {accuracy_soma}")

    #voto
    opinioes = np.array([opiniao_KNN, opiniao_AD, opiniao_NB, opiniao_MLP, opiniao_SVM], dtype=int) #acrescentou o dtype
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
    # opiniao_borda_count = np.argmax(soma_ranks, axis=0) antes
    opiniao_borda_count = (soma_ranks == soma_ranks.min(axis=0)).astype(int) #novo

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
        log_file.write(f"{interation}, Borda Count, {accuracy_borda_count}\n\n")



print("TEMPO TOTAL: ", time.time() - t_init)