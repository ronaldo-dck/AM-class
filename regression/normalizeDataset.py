import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./datasets/house_price_regression_dataset.csv')

columns_to_normalize = ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 
                        'Year_Built', 'Lot_Size', 'Garage_Size', 
                        'Neighborhood_Quality', 'House_Price']

scaler = MinMaxScaler()

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

df.to_csv('./datasets/dataset_normalized.csv', index=False)
