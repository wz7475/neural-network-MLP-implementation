import numpy as np
from keras.utils import to_categorical


class MLP:
    def __init__(self, layers, activation_functions):
        """
        Initializes the MLP.
        layers: a list of integers, where each integer represents the number of neurons in a layer.
        activation_functions: a list of functions, where each function is the activation function for a layer.
        """
        self.layers = layers
        self.activation_functions = activation_functions
        self.num_layers = len(layers)

        self.weights = []
        self.biases = []
        for i in range(1, self.num_layers):
            self.weights.append(np.random.randn(layers[i], layers[i-1]))
            self.biases.append(np.random.randn(layers[i]))

    def forward_pass(self, input_data):
        """
        Propagates the input data through the MLP and returns the output.
        input_data: the input data, which should be a numpy array of shape (batch_size, num_features).
        """
        data = input_data
        for i in range(self.num_layers-1):
            dot_product = np.dot(data, self.weights[i].T) + self.biases[i]
            data = self.activation_functions[i](dot_product)
        return data

    def categorical_cross_entropy_loss(self, y_pred, y_true):
        """
        Calculates the categorical cross-entropy loss between the predicted output and the true output.
        y_pred: the predicted output, which should be a numpy array of shape (batch_size, num_classes).
        y_true: the true output, which should be a numpy array of shape (batch_size, num_classes).
        """
        # Calculate the loss for each sample
        losses = -np.sum(y_true * np.log(y_pred), axis=1)
        # Return the average loss across the batch
        return np.mean(losses)

    def backpropagation(self, y_pred, y_true):
        """
        Adjusts the weights and biases using backpropagation.
        y_pred: the predicted output, which should be a numpy array of shape (batch_size, num_classes).
        y_true: the true output, which should be a numpy array of shape (batch_size, num_classes).
        """
        # Calculate the error at the output layer
        # error = y_pred - y_true
        error = self.categorical_cross_entropy_loss(y_pred, y_true)
        # Iterate backwards through the layers of the MLP
        for i in reversed(range(self.num_layers-1)):
            # Calculate the error at the current layer
            error = error * self.activation_functions[i](y_pred, derivative=True)
            # Calculate the gradient for the weights and biases at the current layer
            gradient_weights = np.dot(error.T, y_pred)
            gradient_biases = np.sum(error, axis=0)
            # Update the weights and biases at the current layer
            learning_rate = 0.1  # You may want to experiment with different learning rates
            self.weights[i] = self.weights[i] - learning_rate * gradient_weights
            self.biases[i] = self.biases[i] - learning_rate * gradient_biases
            # Set the error for the next iteration
            error = np.dot(error, self.weights[i])

    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.1):
        """
        Trains the MLP on the given data using gradient descent.
        X: the input data, which should be a numpy array of shape (num_samples, num_features).
        y: the true labels, which should be a numpy array of shape (num_samples, num_classes).
        epochs: the number of epochs to train for (default: 10).
        batch_size: the size of the mini-batches to use for gradient descent (default: 32).
        learning_rate: the learning rate to use for gradient descent (default: 0.1).
        """
        num_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            shuffle_indices = np.random.permutation(num_samples)
            X = X[shuffle_indices]
            y = y[shuffle_indices]
            # Split the data into mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                # Propagate the input data through the MLP
                y_pred = self.forward_pass(X_batch)
                # Calculate the loss and the gradients
                loss = self.categorical_cross_entropy_loss(y_pred, y_batch)
                self.backpropagation(y_pred, y_batch)
                # Print the loss for each epoch
                print(f"Epoch {epoch+1}/{epochs}, batch {i}/{num_samples}: loss = {loss}")

    def predict(self, X):
        """
        Predicts the output for the given input data.
        X: the input data, which should be a numpy array of shape (num_samples, num_features).
        Returns: a numpy array of shape (num_samples, num_classes) containing the predicted output.
        """
        return self.forward_pass(X)

    def score(self, X, y):
        """
        Calculates the accuracy of the MLP on the given data.
        X: the input data, which should be a numpy array of shape (num_samples, num_features).
        y: the true labels, which should be a numpy array of shape (num_samples,).
        Returns: the accuracy as a float.
        """
        # Predict the output
        y_pred = self.predict(X)
        # Convert the predicted output and the true labels to one-hot encoded form
        y_pred_one_hot = np.zeros_like(y_pred)
        y_pred_one_hot[np.arange(y_pred.shape[0]), np.argmax(y_pred, axis=1)] = 1
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(y_pred.shape[0]), y] = 1
        # Calculate the number of correct predictions
        correct = np.sum(np.all(y_pred_one_hot == y_true_one_hot, axis=1))
        # Calculate the accuracy
        accuracy = correct / y.shape[0]
        return accuracy

def relu(x, derivative=False):
    """
    Calculates the ReLU activation function.
    x: the input data.
    derivative: a boolean flag indicating whether to return the derivative of the function (default: False).
    Returns: the output of the ReLU function.
    """
    if derivative:
        return (x > 0).astype(float)
    return np.maximum(0, x)

def softmax(x, derivative=False):
    """
    Calculates the softmax activation function.
    x: the input data.
    derivative: a boolean flag indicating whether to return the derivative of the function (default: False).
    Returns: the output of the softmax function.
    """
    if derivative:
        # The derivative of the softmax function is not well-defined
        raise ValueError("The derivative of the softmax function is not well-defined")
    # Subtract the maximum value from the input to avoid numerical instability
    x = x - np.max(x, axis=1, keepdims=True)
    # Calculate the exponential of the input
    exp = np.exp(x)
    # Normalize the exponential by dividing by the sum along the class axis
    return exp / np.sum(exp, axis=1, keepdims=True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Load the wine dataset
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = to_categorical(y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)

# Define the number of features and classes
num_features = X.shape[1]
num_classes = len(np.unique(y))

# Define an MLP with 2 hidden layers, each with 128 neurons and ReLU activation
mlp = MLP([num_features, 12, 12, num_classes], [relu, relu, relu, softmax])

mlp.train(X_train, Y_train, epochs=10, batch_size=1, learning_rate=0.1)