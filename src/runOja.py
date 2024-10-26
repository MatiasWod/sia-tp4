import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.Oja import Oja

data = pd.read_csv('./raw_data/europe.csv')
scaler = StandardScaler()
numerical_cols = data.columns[1:]
normalized_data = scaler.fit_transform(data[numerical_cols])
normalized_df = pd.DataFrame(normalized_data, columns=numerical_cols)

input_data_len = normalized_data.shape[1]
iterations = 100
learning_rate = 0.01
random_seed = 42

oja_model = Oja(normalized_data, input_data_len, learning_rate, random_seed=random_seed)

oja_model.train(iterations)

final_weights = oja_model.get_weights()
print("Final weights:", final_weights)


#PCA ANALISIS
pca_evals = oja_model.evaluate(normalized_data)

countries = data.iloc[:, 0]
plt.figure(figsize=(12, 6))
plt.bar(countries, pca_evals)
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('PCA 1')
plt.title('PCA 1 Analysis for Countries with Oja')
plt.show()
