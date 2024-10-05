import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from utils import gerar_arquiteturas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignorar avisos de convergência específicos
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# dados = pd.read_csv("./datasets/diabetes_012_health_indicators_BRFSS2015.csv")
dados = pd.read_csv("./datasets/house_price_regression_dataset.csv")


log_filename = 'logs.csv'


log_knn = 'log_params_knn.csv'
log_mlp = 'log_params_mlp.csv'
log_svm = 'log_params_svm.csv'
log_rf = 'log_params_rf.csv'
log_gb = 'log_params_gb.csv'




t_init = time.time()

for interation in tqdm(range(20), desc='Processing', unit='test'):
    dados = shuffle(dados, random_state=interation)

    X = np.array(dados.iloc[:, :-1])
    Y = np.array(dados.iloc[:, -1])
    
    # Dividindo os dados em treino, validação e teste
    x_treino, x_temp, y_treino, y_temp = train_test_split(
        X, Y, test_size=0.5, random_state=interation)
    
    x_validacao, x_teste, y_validacao, y_teste = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=interation)

    

##########################################################################################################

    # menor = np.inf
    # for j in ("distance", "uniform"):
    #     for i in range(1, 51):
    #         KNN = KNeighborsRegressor(n_neighbors=i, weights=j)
    #         KNN.fit(x_treino, y_treino)
    #         opiniao = KNN.predict(x_validacao)
    #         rmse = np.sqrt(mean_squared_error(y_validacao, opiniao))       
    #         print("K: ", i, " Métrica: ", j, " RMSE: ", rmse)
    #         if (rmse < menor):
    #             menor = rmse
    #             Melhor_k = i
    #             Melhor_metrica = j


    # KNN = KNeighborsRegressor(n_neighbors=Melhor_k,weights=Melhor_metrica)
    # KNN.fit(x_treino,y_treino)
    # opiniao_KNN  = KNN.predict(x_teste)
    
    # rmse_KNN = np.sqrt(mean_squared_error(y_teste, opiniao))       

    # with open(log_knn, 'a+') as log_file:
    #     log_file.write(f'{interation},{Melhor_k},{Melhor_metrica}\n')

###########################################################################################################

    # best_mlp_rmse = np.inf
    # best_i_mlp = 0
    # best_j_mlp = 0
    # best_k_mlp = 0
    # best_func_mlp = 0
    # best_y_pred = 0


    # arquiteturas = gerar_arquiteturas()

    # for arq in arquiteturas:
    #     for learning_rate in ('constant', 'invscaling', 'adaptive'):
    #         for epocas in (100, 200, 300, 400, 500):
    #             for l in ('identity', 'logistic', 'tanh', 'relu'):

    #                 MLP = MLPRegressor(
    #                     hidden_layer_sizes=arq, tol=1e-5, learning_rate=learning_rate, max_iter=epocas, activation=l)
                    
    #                 MLP.fit(x_treino, y_treino)

    #                 opiniao = MLP.predict(x_validacao)
    #                 rmse = np.sqrt(mean_squared_error(y_validacao, opiniao))       

    #                 if (rmse < best_mlp_rmse):
    #                     best_mlp_rmse = rmse
    #                     best_i_mlp = arq
    #                     best_learning_rate_mlp = learning_rate
    #                     best_epocas_mlp = epocas
    #                     best_func_mlp = l
                    


            


    # MLP = MLPRegressor(hidden_layer_sizes=best_i_mlp, learning_rate=best_learning_rate_mlp, max_iter=best_epocas_mlp, activation=best_func_mlp)
    # MLP.fit(x_treino,y_treino)
    # opiniao_MLP = MLP.predict(x_teste)

    # rmse_MLP = np.sqrt(mean_squared_error(y_teste, opiniao_MLP))

    # with open(log_mlp, 'a+') as log_file:
    #     log_file.write(f'{interation},{best_i_mlp},{best_learning_rate_mlp},{best_epocas_mlp},{best_func_mlp}\n')

# ##########################################################################################################
    ## SVM


    # menor = np.inf

    # # Grid search nos parâmetros kernel e C
    # for kernel in ("linear", "poly", "rbf", "sigmoid"):
    #     for C in [0.1, 1, 10, 100]: #retirei o 200
    #         svm = SVR(kernel=kernel, C=C)
    #         svm.fit(x_treino, y_treino)
    #         opiniao = svm.predict(x_validacao)

    #         rmse = np.sqrt(mean_squared_error(y_validacao, opiniao))       
    #         print(f"Kernel: {kernel}, C: {C}, RMSE: {rmse:.2f}")
    #         if (rmse < menor):
    #             menor = rmse
    #             best_kernel = kernel
    #             best_C = C

    # svm = SVR(kernel=best_kernel, C=best_C)
    # svm.fit(x_treino, y_treino)
    # opiniao_SVM = svm.predict(x_teste)
    # rmse_SVM = np.sqrt(mean_squared_error(y_teste, opiniao_SVM))
    # with open(log_svm, 'a+') as log_file:
    #     log_file.write(f'{1},{best_kernel},{best_C}\n')


##############################################################################

    # melhor_rf_rmse = np.inf
    # best_n_estimators = 0
    # best_criterion = ''
    # best_max_depth = 0
    # best_min_samples_split = 0
    # best_min_samples_leaf = 0

    # for n_estimators in [50, 100, 150, 200]:
    #     print(time.time())
    #     for criterion in ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']:
    #         for max_depth in [10, 20, 30]:
    #             for min_samples_split in [2, 5, 10]:
    #                 for min_samples_leaf in [1, 2, 4]:
    #                     rf = RandomForestRegressor(n_estimators=n_estimators, 
    #                                                 criterion=criterion,
    #                                                 max_depth=max_depth,
    #                                                 min_samples_split=min_samples_split,
    #                                                 min_samples_leaf=min_samples_leaf,
    #                                                 random_state=42)

    #                     rf.fit(x_treino, y_treino)
    #                     opiniao = rf.predict(x_validacao)

    #                     rmse_rf = np.sqrt(mean_squared_error(y_validacao, opiniao))

    #                     if rmse_rf < melhor_rf_rmse:
    #                         melhor_rf_rmse = rmse_rf
    #                         best_n_estimators = n_estimators
    #                         best_criterion = criterion
    #                         best_max_depth = max_depth
    #                         best_min_samples_split = min_samples_split
    #                         best_min_samples_leaf = min_samples_leaf

    # rf_final = RandomForestRegressor(n_estimators=best_n_estimators, 
    #                                 criterion=best_criterion,
    #                                 max_depth=best_max_depth,
    #                                 min_samples_split=best_min_samples_split,
    #                                 min_samples_leaf=best_min_samples_leaf,
    #                                 random_state=42)

    # rf_final.fit(x_treino, y_treino)
    # opiniao_rf = rf_final.predict(x_teste)
    # rmse_rf_final = np.sqrt(mean_squared_error(y_teste, opiniao_rf))

    # with open(log_rf, 'a+') as log_file:
    #     log_file.write(f'{interation},{best_n_estimators},{best_criterion},{best_max_depth},{best_min_samples_split},{best_min_samples_leaf}\n')


###########################################################################################################
    melhor_gb_rmse = np.inf
    best_n_estimators_gb = 0
    best_loss = ''
    best_max_depth_gb = 0
    best_learning_rate = 0
    best_min_samples_split_gb = 0
    best_min_samples_leaf_gb = 0

    for n_estimators in range(100, 1501, 100): # pelo que eu li é melhor ser um valor maior, GB consegue trabalhar bem contra o overfitting
        print(time.time() - t_init)
        for loss in ["squared_error", "absolute_error", "quantile"]:
            for max_depth in [3, 5, 10]:  # Depth padrão para GB é geralmente menor que RF
                for learning_rate in (0.001, 0.01, 0.1):
                    for min_samples_split in [2, 5]:
                        for min_samples_leaf in [1, 2, 4]:

                            gb = GradientBoostingRegressor(n_estimators=n_estimators,
                                                            loss=loss,
                                                            max_depth=max_depth,
                                                            learning_rate=learning_rate,
                                                            min_samples_split=min_samples_split,
                                                            min_samples_leaf=min_samples_leaf,
                                                            random_state=42)

                            gb.fit(x_treino, y_treino)
                            opiniao = gb.predict(x_validacao)

                            rmse_gb = np.sqrt(mean_squared_error(y_validacao, opiniao))

                            if rmse_gb < melhor_gb_rmse:
                                melhor_gb_rmse = rmse_gb
                                best_n_estimators_gb = n_estimators
                                best_loss = loss
                                best_max_depth_gb = max_depth
                                best_learning_rate = learning_rate
                                best_min_samples_split_gb = min_samples_split
                                best_min_samples_leaf_gb = min_samples_leaf

    gb_final = GradientBoostingRegressor(n_estimators=best_n_estimators_gb,
                                        loss=best_loss,
                                        max_depth=best_max_depth_gb,
                                        learning_rate=best_learning_rate,
                                        min_samples_split=best_min_samples_split_gb,
                                        min_samples_leaf=best_min_samples_leaf_gb,
                                        random_state=42)

    gb_final.fit(x_treino, y_treino)
    opiniao_gb = gb_final.predict(x_teste)
    rmse_gb_final = np.sqrt(mean_squared_error(y_teste, opiniao_gb))

    with open(log_gb, 'a+') as log_file:
        log_file.write(f'{interation},{best_n_estimators_gb},{best_loss},{best_max_depth_gb},{best_learning_rate},{best_min_samples_split_gb},{best_min_samples_leaf_gb}\n')



###########################################################################################################

    rlm_final = LinearRegression()

    rlm_final.fit(x_treino, y_treino)
    opiniao_rlm = rlm_final.predict(x_teste)
    rmse_rlm_final = np.sqrt(mean_squared_error(y_teste, opiniao_rlm))



    with open(log_filename, 'a+') as log_file:
        log_file.write(f"{interation}, KNR, {rmse_KNN}\n")
        log_file.write(f"{interation}, MLP, {rmse_MLP}\n")
        log_file.write(f"{interation}, SVM, {rmse_SVM}\n")
        log_file.write(f"{interation}, RF, {rmse_rf_final}\n")
        log_file.write(f"{interation}, GB, {rmse_gb_final}\n")
        log_file.write(f"{interation}, RLM, {rmse_rlm_final}\n")
        



print("TEMPO TOTAL: ", time.time() - t_init)