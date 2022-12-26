import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        # Initialize weights and biases to small random values
        self.weights = np.random.rand(X.shape[1])
        self.biases = np.random.rand(X.shape[1])

        # Loop through the number of iterations
        for i in range(self.num_iterations):
            # Loop through each example in the training set
            for x, label in zip(X, y):
                # Make a prediction using the current weights and biases
                prediction = self.predict(x)

                # Update the weights and biases if the prediction was incorrect
                if prediction != label:
                    error = label - prediction
                    self.weights += self.learning_rate * error * x
                    self.biases += self.learning_rate * error

    def predict(self, x):
        # Calculate the dot product of the input features and the weights
        dot_product = np.dot(x, self.weights) + np.dot(x, self.biases)

        # Return 1 if the dot product is positive, otherwise return 0
        return 1 if dot_product >= 0 else 0
