import matplotlib.pyplot as plt
from src.Hopfield import HopfieldNetwork
from src.Utils.alphabet import read_matrices_from_file, print_letters_line

# Initialize the Hopfield network
network = HopfieldNetwork()

filename = './raw_data/alphabet.txt'
matrices = read_matrices_from_file(filename)

letter_g = matrices[6]
noisy_g = network.add_noise(letter_g, 0.2)

print_letters_line([letter_g, noisy_g])