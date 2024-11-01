import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from src.Models.Kohonen import Kohonen
import random

# Cargar el dataset desde el archivo CSV
data = pd.read_csv('./raw_data/europe.csv')

# Seleccionar las columnas numéricas
features = data[['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']]

# Estandarizar los datos
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Definir los valores de k, sigma y learning_rate
k_values = [4]  # Diferentes tamaños de mapa
sigma_values = [3.0]  # Diferentes valores de sigma
learning_rates = [0.1]  # Diferentes tasas de aprendizaje

# Función para calcular y graficar la matriz U
def calculate_u_matrix(som):
    weights = som.get_weights()
    u_matrix = np.zeros((som.map_size[0], som.map_size[1]))

    for i in range(som.map_size[0]):
        for j in range(som.map_size[1]):
            neighbors = []
            if i > 0: neighbors.append(weights[i - 1, j])  # Arriba
            if i < som.map_size[0] - 1: neighbors.append(weights[i + 1, j])  # Abajo
            if j > 0: neighbors.append(weights[i, j - 1])  # Izquierda
            if j < som.map_size[1] - 1: neighbors.append(weights[i, j + 1])  # Derecha

            distances = [np.linalg.norm(weights[i, j] - neighbor) for neighbor in neighbors]
            u_matrix[i, j] = np.mean(distances)
    return u_matrix

# Función para generar colores aleatorios
def random_color():
    return (random.random(), random.random(), random.random())

for k in k_values:
    for sigma in sigma_values:
        for lr in learning_rates:
            map_size = (k, k)
            input_dim = standardized_features.shape[1]

            # Inicializar y entrenar el modelo
            som = Kohonen(
                input_dim=input_dim,
                map_size=map_size,
                learning_rate=lr,
                sigma=sigma,
                random_seed=41
            )
            som.train(standardized_features, epochs=500 * input_dim)

            # Obtener posiciones BMU para cada país
            bmu_coordinates = som.transform(standardized_features)
            data['BMU'] = [tuple(coord) for coord in bmu_coordinates]

            # Crear un diccionario para almacenar los países en cada celda
            bmu_dict = {}
            for _, row in data.iterrows():
                bmu = row['BMU']
                country = row['Country']
                if bmu not in bmu_dict:
                    bmu_dict[bmu] = []
                bmu_dict[bmu].append(country)

            # Gráfico del Mapa de Países Agrupados
            fig, ax = plt.subplots(figsize=(8, 8))
            cell_colors = {}

            for i in range(map_size[0]):
                for j in range(map_size[1]):
                    countries_in_cell = bmu_dict.get((i, j), [])
                    if (i, j) not in cell_colors:
                        cell_colors[(i, j)] = random_color()

                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, color=cell_colors[(i, j)], alpha=0.3)
                    ax.add_patch(rect)

                    text = "\n".join(countries_in_cell)
                    ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black')

            ax.set_xlim(-0.5, map_size[1] - 0.5)
            ax.set_ylim(-0.5, map_size[0] - 0.5)
            ax.set_xticks(np.arange(map_size[1]))
            ax.set_yticks(np.arange(map_size[0]))
            ax.invert_yaxis()
            plt.title(f'Kohonen - Mapa de Países para k={k}, sigma={sigma}, lr={lr}')
            plt.show()

            # Gráfico de la matriz U (distancias promedio)
            u_matrix = calculate_u_matrix(som)
            plt.figure(figsize=(8, 8))
            plt.imshow(u_matrix, cmap='Blues', interpolation='nearest')
            plt.colorbar(label='Distancia promedio')
            plt.title(f'Kohonen - Distancias Vecinas para k={k}, sigma={sigma}, lr={lr}')
            plt.xticks(np.arange(map_size[1]), np.arange(map_size[1]))
            plt.yticks(np.arange(map_size[0]), np.arange(map_size[0]))

            for i in range(map_size[0]):
                for j in range(map_size[1]):
                    plt.text(j, i, f"{u_matrix[i, j]:.2f}", ha='center', va='center', color='black')

            plt.show()

            # Gráfico de la frecuencia de selección de BMU
            bmu_count = som.get_bmu_counts()
            plt.figure(figsize=(8, 8))
            plt.imshow(bmu_count, cmap='Blues', interpolation='nearest')
            plt.colorbar(label='BMU Count')
            plt.title(f'Kohonen - Frecuencia de BMU para k={k}, sigma={sigma}, lr={lr}')
            plt.xticks(np.arange(bmu_count.shape[1]), np.arange(bmu_count.shape[1]))
            plt.yticks(np.arange(bmu_count.shape[0]), np.arange(bmu_count.shape[0]))

            for i in range(bmu_count.shape[0]):
                for j in range(bmu_count.shape[1]):
                    plt.text(j, i, f"{bmu_count[i, j]}", ha='center', va='center', color='black')

            plt.show()

            for idx, column in enumerate(features.columns):
                variable_weights = som.get_weights()[:, :, idx]
                plt.figure(figsize=(8, 8))
                plt.imshow(variable_weights, cmap='viridis', interpolation='nearest')
                plt.colorbar(label=f'Peso')
                plt.title(f'Kohonen - Influencia de {column} para k={k}, sigma={sigma}, lr={lr}')
                plt.xticks(np.arange(map_size[1]), np.arange(map_size[1]))
                plt.yticks(np.arange(map_size[0]), np.arange(map_size[0]))

                for i in range(map_size[0]):
                    for j in range(map_size[1]):
                        plt.text(j, i, f"{variable_weights[i, j]:.2f}", ha='center', va='center', color='black')
                plt.show()
