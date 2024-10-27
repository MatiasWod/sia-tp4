import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def create_letter_plot(letter, ax, cmap='Blues'):
    p = sns.heatmap(letter, ax=ax, annot=False, cbar=False, cmap=cmap, square=True, linewidth=2, linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p

def print_letters_line(letters, cmap='Blues', cmaps=[]):
    fig, ax = plt.subplots(1, len(letters))
    fig.set_dpi(360)
    if not cmaps:
        cmaps = [cmap]*len(letters)
    if len(cmaps) != len(letters):
        raise Exception('cmap list should be the same lenght as letters')
    for i, subplot in enumerate(ax):
        create_letter_plot(letters[i].reshape(5,5), ax=subplot, cmap=cmaps[i])
    plt.show()