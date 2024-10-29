import json
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.Models.Oja import Oja

data = pd.read_csv('./raw_data/europe.csv')
scaler = StandardScaler()
numerical_cols = data.columns[1:]
normalized_data = scaler.fit_transform(data[numerical_cols])
normalized_df = pd.DataFrame(normalized_data, columns=numerical_cols)

input_data_len = normalized_data.shape[1]


with open("./configs/ej2.json") as file:
    config = json.load(file)

    iterations = config["iterations"]
    learning_rate = config['learning_rate']
    random_seed = 42

    oja_model = Oja(normalized_data, input_data_len, learning_rate, random_seed=random_seed)

    oja_model.train(iterations)

    final_weights = oja_model.get_weights()
    print("Final weights:", final_weights)


    #PCA ANALISIS
    pca_evals = oja_model.evaluate(normalized_data)

    countries = data.iloc[:, 0]
    plt.figure(figsize=(12, 8))
    bars = plt.bar(countries, pca_evals)
    plt.xticks(rotation=90, fontsize=20)
    plt.xlabel('Country', fontsize=20)
    plt.ylabel('PC 1', fontsize=20)
    plt.title('PC 1 Analysis for Countries with Oja', fontsize=20)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                 ha='center', va='bottom', fontsize=12, color="black")

    plt.show()