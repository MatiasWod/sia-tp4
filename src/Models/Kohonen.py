import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Kohonen:
    def __init__(self, input_dim, map_size=(2, 2), learning_rate=0.5, sigma=1.0, random_seed=None):
        """
        Args:
        input_dim : int
            Dimension of input data
        map_size : tuple
            (height, width)
        learning_rate : float
            Initial learning rate
        sigma : float
            Initial neighborhood radius
        random_seed : int
            Seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.input_dim = input_dim
        self.map_size = map_size
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.initial_sigma = sigma
        self.sigma = sigma

        self.bmu_count = np.zeros(self.map_size, dtype=int)

        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)
        self.neuron_positions = np.array([[(i, j) for j in range(map_size[1])]
                                          for i in range(map_size[0])])

    def inverse_transform(self, data):
        """
        Convert normalized data back to original scale

        Args:
        data : numpy.array
            Normalized data

        Returns:
        numpy.array
            Data in original scale
        """
        return self.scaler.inverse_transform(data)

    def find_bmu(self, input_vector):
        """
        Find the Best Matching Unit (BMU) for the input vector

        Args:
        input_vector : numpy.array
            Input data vector

        Returns:
        tuple
            (row, col) position of the BMU
        """
        input_reshaped = input_vector.reshape(1, 1, -1)

        # Calculate Euclidean distance between input and all neurons
        distances = np.sqrt(np.sum((self.weights - input_reshaped) ** 2, axis=2))

        # Find position of the neuron with minimum distance
        bmu_pos = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_pos

    def update_weights(self, input_vector, bmu_pos, iteration, total_iterations):
        """
        Update weights based on BMU and neighborhood function

        Args:
        input_vector : numpy.array
            Input data vector
        bmu_pos : tuple
            Position of the Best Matching Unit
        iteration : int
            Current iteration number
        total_iterations : int
            Total number of iterations
        """
        # Update learning rate and sigma based on iteration
        self.learning_rate = self._decay_learning_rate(iteration, total_iterations)
        self.sigma = self._decay_sigma(iteration, total_iterations)

        # Calculate distance of all neurons to BMU
        bmu_distance = np.sqrt(np.sum((self.neuron_positions - np.array(bmu_pos)) ** 2, axis=2))

        # Calculate neighborhood function
        neighborhood = np.exp(-bmu_distance ** 2 / (2 * self.sigma ** 2))

        # Reshape for broadcasting
        neighborhood = neighborhood.reshape(self.map_size[0], self.map_size[1], 1)
        input_reshaped = input_vector.reshape(1, 1, -1)

        # Update weights
        self.weights += self.learning_rate * neighborhood * (input_reshaped - self.weights)

    def _decay_learning_rate(self, iteration, total_iterations):
        """Calculate learning rate decay"""
        return self.initial_learning_rate * np.exp(-iteration / total_iterations)

    def _decay_sigma(self, iteration, total_iterations):
        """Calculate sigma (neighborhood radius) decay"""
        return self.initial_sigma * np.exp(-iteration / total_iterations)

    def train(self, normalized_data, epochs=100):
        """
        Train the SOM on the input data

        Args:
        normalized_data : numpy.array
            Training data array (n_samples, input_dim)
        epochs : int
            Number of training epochs
        """

        total_iterations = epochs * len(normalized_data)
        current_iteration = 0

        for epoch in range(epochs):
            for input_vector in normalized_data:
                # Find best matching unit
                bmu_pos = self.find_bmu(input_vector)

                self.bmu_count[bmu_pos] += 1

                # Update weights
                self.update_weights(input_vector, bmu_pos, current_iteration, total_iterations)

                current_iteration += 1

    def transform(self, normalized_data):
        """
        Transform input data to BMU coordinates

        Args:
        normalized_data : numpy.array
            Input data array (n_samples, input_dim)

        Returns:
        numpy.array
            Array of BMU coordinates for each input sample
        """

        bmu_coordinates = np.zeros((len(normalized_data), 2))
        for i, vector in enumerate(normalized_data):
            bmu_coordinates[i] = self.find_bmu(vector)
        return bmu_coordinates

    def get_weights(self):
        """Return the weights of the SOM"""
        return self.weights.copy()

    def get_original_weights(self):
        """Return the weights in original scale"""
        weights_shape = self.weights.shape
        weights_2d = self.weights.reshape(-1, self.input_dim)
        original_weights_2d = self.inverse_transform(weights_2d)
        return original_weights_2d.reshape(weights_shape)

    def get_bmu_counts(self):
        return self.bmu_count