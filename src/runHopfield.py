import numpy as np
from src.Models.Hopfield import HopfieldNetwork

def read_matrices_from_file(filename):
    matrices = []
    current_matrix = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [int(num) for num in line.split(',')]
                current_matrix.append(row)

                if len(current_matrix) == 5:
                    matrices.append(np.array(current_matrix))
                    current_matrix = []

    return matrices

# Usage
filename = './raw_data/alphabet.txt'
matrices = read_matrices_from_file(filename)

# Example: Print each matrix
for i, matrix in enumerate(matrices):
    print(f"Letter {chr(65 + i)}:\n{matrix}\n")

hopfield = HopfieldNetwork()
hopfield.train(matrices[3:8])
print(f"longitud del alphabet: {len(matrices[3:7])}")
original_matrix = matrices[3]
noisy_matrix = hopfield.add_noise(original_matrix,0.1)
hopfield_matrix = hopfield.update(noisy_matrix)
print("Should print V")
print("Original matrix")
print(original_matrix)
print("-------------------------")
print("noisy_matrix")
print(noisy_matrix)
print("-------------------------")
print("hopfield prediction")
print(hopfield_matrix)