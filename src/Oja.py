import numpy as np


class Oja:
    def __init__(self, input_data, input_data_len, iterations, learning_rate, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        self.input_data = input_data
        self.iterations = iterations
        self.initial_learning_rate = learning_rate
        self.weights = np.random.rand(input_data_len)

    def train(self):
        for index, _ in enumerate(range(self.iterations)):
            current_lr = self.initial_learning_rate / (index + 1)
            for data in self.input_data:
                out = np.dot(data, self.weights)
                dw = current_lr * out * ( data - out * self.weights)
                self.weights += dw

    def evaluate(self, input_data):
        evaluation = [np.dot(data, self.weights) for data in input_data]
        return evaluation

    def get_weights(self):
        return self.weights
