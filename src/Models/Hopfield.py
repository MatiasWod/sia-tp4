import numpy as np


class HopfieldNetwork:
    def __init__(self, shape=(5, 5)):
        # Pattern matrix shape, in our case (5x5)
        self.shape = shape
        self.size = shape[0] * shape[1]
        self.weights = np.zeros(shape + shape)

    def train(self, patterns):
        self.weights = np.zeros(self.shape + self.shape)

        for pattern in patterns:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[0]):
                        for l in range(self.shape[1]):
                            if (i, j) != (k, l):
                                self.weights[i, j, k, l] += pattern[i, j] * pattern[k, l]

        self.weights /= len(patterns)

    def update(self, pattern, max_iterations=20):
        current_state = pattern.copy()

        for _ in range(max_iterations):
            prev_state = current_state.copy()

            positions = [(i, j) for i in range(self.shape[0])
                         for j in range(self.shape[1])]
            np.random.shuffle(positions)

            for i, j in positions:
                activation = 0
                for k in range(self.shape[0]):
                    for l in range(self.shape[1]):
                        activation += self.weights[i, j, k, l] * current_state[k, l]

                current_state[i, j] = 1 if activation > 0 else -1

            if np.array_equal(prev_state, current_state):
                break

        return current_state

    def add_noise(self, pattern, noise_level=0.1):
        noisy_pattern = pattern.copy()
        nums_to_change= int(self.size * noise_level)


        possible_coordinates = [(i, j) for i in range(self.shape[0])
                     for j in range(self.shape[1])]

        flip_positions = np.random.choice(len(possible_coordinates), nums_to_change, replace=False)

        for idx in flip_positions:
            i, j = possible_coordinates[idx]
            noisy_pattern[i, j] *= -1

        return noisy_pattern
