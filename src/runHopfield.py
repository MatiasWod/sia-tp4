import numpy as np
from src.Hopfield import HopfieldNetwork

letters = [np.array([[ 1,  1,  1,  1, 1],
              [ 1, -1, -1, -1,  1],
              [ 1,  1,  1,  1, -1],
              [ 1, -1, -1, -1,  1],
              [ 1,  1,  1,  1, 1]]),
           np.array([[1, 1, 1, 1, -1],
                     [1, -1, -1, 1, -1],
                     [1, -1, -1, 1, -1],
                     [1, 1, 1, 1, -1],
                     [-1, -1, -1, -1, 1]]),
           np.array([[1, 1, 1, 1, 1],
                     [-1, -1, 1, -1, -1],
                     [-1, -1, 1, -1, -1],
                     [-1, -1, 1, -1, -1],
                     [-1, -1, 1, -1, -1]]),
           np.array([[1, -1, -1, -1, 1],
                     [1, -1, -1, -1, 1],
                     [-1, 1, -1, 1, -1],
                     [-1, 1, -1, 1, -1],
                     [-1, -1, 1, -1, -1]])
]

hopfield = HopfieldNetwork()
hopfield.train(letters)
original_matrix = letters[3]
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