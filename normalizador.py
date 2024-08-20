import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Carregar os dados
              


df = pd.read_csv("./datasets/diabetes_012_health_indicators_BRFSS2015.csv")

# Separar a coluna de classe (Diabetes_012) das demais
classe = df['Diabetes_012']
features = df.drop(columns=['Diabetes_012'])

# Aplicar a normalização (Min-Max Scaling) nas features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Converter de volta para DataFrame e adicionar a coluna de classe novamente
df_normalized = pd.DataFrame(features_normalized, columns=features.columns)
df_normalized['Diabetes_012'] = classe.values

# Salvar a base de dados normalizada
df_normalized.to_csv('base_de_dados_normalizada.csv', index=False)

print("Base de dados normalizada e salva com sucesso!")
