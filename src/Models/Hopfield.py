import numpy as np

class HopfieldNetwork:
    def __init__(self, shape=(5, 5)):
        self.shape = shape  # Shape of the pattern matrix (e.g., 5x5)
        self.size = shape[0] * shape[1]  # Total number of elements in the matrix
        self.weights = np.zeros(shape + shape)  # Weight matrix

    def train(self, patterns):
        """Train the network using Hebbian learning."""
        self.weights = np.zeros(self.shape + self.shape)

        for pattern in patterns:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[0]):
                        for l in range(self.shape[1]):
                            if (i, j) != (k, l):
                                self.weights[i, j, k, l] += pattern[i, j] * pattern[k, l]

        self.weights /= len(patterns)  # Normalize by the number of patterns

    def energy(self, pattern):
        """Calculate the energy of a given pattern."""
        energy = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[0]):
                    for l in range(self.shape[1]):
                        energy -= self.weights[i, j, k, l] * pattern[i, j] * pattern[k, l]
        return energy / 2  # Factor 1/2 to avoid double counting

    def update(self, pattern, max_iterations=20):
        """Update the network state and track energy and state at each step."""
        current_state = pattern.copy()
        energies = []  # Track energy at each epoch
        state_evolution = []  # Track the state at each step

        for epoch in range(max_iterations):
            prev_state = current_state.copy()

            # Record the energy and current state for plotting
            energies.append(self.energy(current_state))
            state_evolution.append(current_state.copy())

            # Randomize the update order of the neurons
            positions = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]
            np.random.shuffle(positions)

            # Update each neuron asynchronously
            for i, j in positions:
                activation = 0
                for k in range(self.shape[0]):
                    for l in range(self.shape[1]):
                        activation += self.weights[i, j, k, l] * current_state[k, l]
                
                current_state[i, j] = 1 if activation > 0 else -1

            if np.array_equal(prev_state, current_state):  # Stop if the state stabilizes
                break

        # Append the final energy and state
        energies.append(self.energy(current_state))
        state_evolution.append(current_state)

        return current_state, energies, state_evolution

    def add_noise(self, pattern, noise_level=0.1):
        """Add noise to a pattern by flipping a fraction of its elements."""
        noisy_pattern = pattern.copy()
        nums_to_change = int(self.size * noise_level)

        possible_coordinates = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]
        flip_positions = np.random.choice(len(possible_coordinates), nums_to_change, replace=False)

        for idx in flip_positions:
            i, j = possible_coordinates[idx]
            noisy_pattern[i, j] *= -1

        return noisy_pattern
