import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("./datasets/diabetes_012_health_indicators_BRFSS2015.csv")

classe = df['Diabetes_012']
features = df.drop(columns=['Diabetes_012'])

scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

df_normalized = pd.DataFrame(features_normalized, columns=features.columns)
df_normalized['Diabetes_012'] = classe.values

df_normalized.to_csv('base_de_dados_normalizada.csv', index=False)

print("Base de dados normalizada e salva com sucesso!")
