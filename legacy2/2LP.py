import numpy as np


class Perceptron:
    def __init__(self, n_inputs, n_hidden, learning_rate):
        # Initialize the hidden layer weights to random values
        self.hidden_weights = np.random.rand(n_inputs, n_hidden)

        # Initialize the hidden layer biases to random values
        self.hidden_biases = np.random.rand(n_hidden)

        # Initialize the output layer weights to random values
        self.output_weights = np.random.rand(n_hidden)

        # Initialize the output layer biases to random values
        self.output_biases = np.random.rand(1)

        # Initialize the learning rate
        self.learning_rate = learning_rate

    def predict(self, x):
        # Calculate the dot product of the input features and the hidden layer weights
        hidden_inputs = np.dot(x, self.hidden_weights)

        # Add the hidden layer biases
        hidden_inputs += self.hidden_biases

        # Apply the sigmoid activation function to the hidden layer inputs
        self.hidden_outputs = 1 / (1 + np.exp(-hidden_inputs))

        # Calculate the derivative of the hidden layer outputs with respect to the hidden layer inputs
        # TODO make attribute instead of local variable
        hidden_outputs_derivative = self.hidden_outputs * (1 - self.hidden_outputs)

        # Calculate the dot product of the hidden layer outputs and the output layer weights
        output = np.dot(self.hidden_outputs, self.output_weights)

        # Add the output layer bias
        output += self.output_biases

        # Return 1 if the output is positive, otherwise return 0
        return 1 if output > 0 else 0

    def fit(self, X, y, epochs):
        # Loop over the specified number of epochs
        for epoch in range(epochs):
            # Loop over the training examples
            for x, label in zip(X, y):
                # Make a prediction on the training example
                prediction = self.predict(x)

                # Calculate the error between the prediction and the true label
                error = label - prediction

                # Calculate the gradient of the error with respect to the hidden layer weights
                error_gradient_hidden_weights = error * hidden_outputs_derivative * x.reshape(-1, 1)

                # Calculate the gradient of the error with respect to the hidden layer biases
                error_gradient_hidden_biases = error * hidden_outputs_derivative

                # Calculate the gradient of the error with respect to the output layer weights
                error_gradient_output_weights = error * self.hidden_outputs

                # Calculate the gradient of the error with respect to the output layer biases
                error_gradient_output_biases = error

                # Update the hidden layer weights
                self.hidden_weights += self.learning_rate * error_gradient_hidden_weights

                # Update the hidden layer biases
                self.hidden_biases += self.learning_rate * error_gradient_hidden_biases

                # Update the output layer weights
                self.output_weights += self.learning_rate * error_gradient
