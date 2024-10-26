import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.Models.Kohonen import Kohonen

# Cargar el dataset desde el archivo CSV
data = pd.read_csv('./raw_data/europe.csv')

# Seleccionar las columnas numéricas (No se pueden procesar datos no numéricos)
features = data[['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']]

# Estandarizar los datos
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Inicializar y entrenar el modelo
input_dim = standardized_features.shape[1]
map_size = (3, 3)

som = Kohonen(
    input_dim=input_dim, 
    map_size=map_size, 
    learning_rate=0.1, 
    sigma=3.0, 
    normalization='zscore', 
    random_seed=41
)
som.train(standardized_features, epochs=500*input_dim)

# 5. Obtener las posiciones BMU para cada país
bmu_coordinates = som.transform(standardized_features)
data['BMU'] = [tuple(coord) for coord in bmu_coordinates]

# Crear un diccionario para almacenar los países en cada celda porque se pueden superponer
bmu_dict = {}
for _, row in data.iterrows():
    bmu = row['BMU']
    country = row['Country']
    if bmu not in bmu_dict:
        bmu_dict[bmu] = []
    bmu_dict[bmu].append(country)

# Gráfico del Mapa con los países enlistados en cada coordenada
fig, ax = plt.subplots(figsize=(8, 8))

# Graficar el mapa
for i in range(map_size[0]):
    for j in range(map_size[1]):
        # Obtener los países de la coordenada actual
        countries_in_cell = bmu_dict.get((i, j), [])
        text = "\n".join(countries_in_cell)  # Unir los nombres con saltos de línea
        ax.text(j, i, text, ha='center', va='center', fontsize=8)

# Configurar límites y estética del gráfico
ax.set_xlim(-0.5, map_size[1] - 0.5)  # Limites x
ax.set_ylim(-0.5, map_size[0] - 0.5)  # Limites y

ax.set_xticks(np.arange(map_size[1]))  # Marcas en x desde 0 hasta map_size[1]-1
ax.set_yticks(np.arange(map_size[0]))  # Marcas en y desde 0 hasta map_size[0]-1
ax.set_xticklabels(np.arange(map_size[1]))  # Etiquetas en x
ax.set_yticklabels(np.arange(map_size[0]))  # Etiquetas en y
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
ax.invert_yaxis()  # Invertir el eje Y para alinear correctamente

plt.title('SOM - Mapa de Países Agrupados')
plt.show()

# Gráfico de la matriz de distancias promedio entre neuronas vecinas
def calculate_u_matrix(som):
    """Calcula la U-Matrix (distancia promedio entre neuronas vecinas)."""
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

u_matrix = calculate_u_matrix(som)

# Gráfico de la de distancias vecinas
plt.figure(figsize=(8, 8))
plt.imshow(u_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Distancia promedio')
plt.title('Mapa U - Distancias entre Neuronas Vecinas')

# Configurar límites y marcas del gráfico
plt.xticks(np.arange(map_size[1]), np.arange(map_size[1]))  # Marcas en x desde 0 hasta map_size[1]-1
plt.yticks(np.arange(map_size[0]), np.arange(map_size[0]))  # Marcas en y desde 0 hasta map_size[0]-1

plt.show()

#Grafico para ver las neuronas muertas
bmu_count = som.get_bmu_counts()
plt.figure(figsize=(8, 8))
plt.imshow(bmu_count, cmap='Blues', interpolation='nearest')
plt.colorbar(label='BMU Count')
plt.title('Frequency of BMU Selection')

plt.xticks(np.arange(bmu_count.shape[1]), np.arange(bmu_count.shape[1]))
plt.yticks(np.arange(bmu_count.shape[0]), np.arange(bmu_count.shape[0]))

plt.show()
