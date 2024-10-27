import numpy as np
from src.Hopfield import HopfieldNetwork
from src.Utils.alphabet import read_matrices_from_file
from src.Utils.alphabet import print_letters_line, read_matrices_from_file

filename = './raw_data/alphabet.txt'
matrices = read_matrices_from_file(filename)

print_letters_line(matrices[:5])
print_letters_line(matrices[5:10])
print_letters_line(matrices[10:15])
print_letters_line(matrices[15:20])
print_letters_line(matrices[20:26])