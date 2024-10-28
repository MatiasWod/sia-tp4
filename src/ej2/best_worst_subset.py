import numpy as np
import pandas as pd
from itertools import combinations
from src.Utils.alphabet import read_matrices_from_file

filename = './raw_data/alphabet.txt'
matrices = read_matrices_from_file(filename)

def calcular_metricas(subset, i=0):
    """Calcula el promedio, máximo producto escalar y su frecuencia para un subconjunto de matrices."""
    n = len(subset)
    resultado = np.zeros((n, n))

    # Calcular los productos internos y llenar la matriz
    for i in range(n):
        for j in range(n):
            if i != j:
                resultado[i, j] = np.dot(subset[i].flatten(), subset[j].flatten())

    resultado_abs = np.abs(resultado)
    # Calcular promedio
    suma_productos = np.sum(resultado_abs)
    num_elementos = resultado_abs.size - n  # Total elementos fuera de la diagonal
    promedio = suma_productos / num_elementos
    
    # Calcular máximo y su frecuencia
    max_producto = np.max(resultado_abs)  # Máximo valor en la matriz
    frecuencia_max = np.count_nonzero(resultado_abs == max_producto)/2  # Frecuencia del máximo
    if promedio == 1:
        print(resultado)
        print(max_producto)
        print(frecuencia_max)
        print(resultado_abs)

    return promedio, max_producto, frecuencia_max

# Lista para almacenar los resultados de cada combinación
resultados = []

# Probar todas las combinaciones de 4 matrices
for indices in combinations(range(26), 4):
    subset = [matrices[i] for i in indices]
    promedio_actual, max_producto_actual, frecuencia_max_actual = calcular_metricas(subset)

    # Convertir los índices a letras
    letras = [chr(65 + i) for i in indices]

    # Guardar los resultados
    resultados.append({
        "Promedio": promedio_actual,
        "Máximo": max_producto_actual,
        "Frecuencia del Máximo": frecuencia_max_actual,
        "Combinación": ", ".join(letras)
    })

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)

# Obtener las 15 combinaciones con menor promedio
top_15_menores = df_resultados.nsmallest(15, 'Promedio')

# Obtener las 15 combinaciones con mayor promedio
top_15_menores = df_resultados.nlargest(15, 'Promedio')

# Mostrar los resultados
print(top_15_menores)

# Guardar el DataFrame como archivo CSV
df_resultados.to_csv("./results/ej2/resultados_combinaciones.csv", index=False)

print("Resultados guardados en 'resultados_combinaciones.csv'")
