
import numpy as np


class Perceptron:
    def __init__(self, num_features, num_hidden_units, num_classes):
        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes
        self.weights_input_to_hidden = np.zeros((num_hidden_units, num_features))
        self.weights_hidden_to_output = np.zeros((num_classes, num_hidden_units))
        self.bias_hidden = np.zeros(num_hidden_units)
        self.bias_output = np.zeros(num_classes)

    def predict(self, x):
        # Calculate activations of hidden units
        hidden_activations = np.dot(self.weights_input_to_hidden, x) + self.bias_hidden
        # Apply activation function to hidden activations
        hidden_output = self._sigmoid(hidden_activations)
        # Calculate activations of output units
        output_activations = np.dot(self.weights_hidden_to_output, hidden_output) + self.bias_output
        # Apply activation function to output activations
        output = self._softmax(output_activations)
        return output

    def train(self, x, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for i in range(len(x)):
                prediction = self.predict(x[i])
                error = y[i] - prediction
                # Calculate activations of hidden units
                hidden_activations = np.dot(self.weights_input_to_hidden, x[i]) + self.bias_hidden
                # Apply activation function to hidden activations
                hidden_output = self._sigmoid(hidden_activations)
                # Update weights and biases
                self.weights_hidden_to_output += (learning_rate * error * hidden_output[:, np.newaxis]).reshape(self.num_classes, self.num_hidden_units)
                self.bias_output += learning_rate * error
                self.weights_input_to_hidden += np.kron(learning_rate * hidden_output * (1 - hidden_output), x[i]).reshape(self.num_hidden_units, self.num_features)
                self.bias_hidden += learning_rate * hidden_output * (1 - hidden_output)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        return (np.exp(x) / np.exp(x).sum())


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load the Wine dataset
X, y = datasets.load_iris(return_X_y=True)
y_cat = []
converter = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1],
}
for label in y:
    y_cat.append(converter[label])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)


# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore')
# y_hot = enc.fit_transform(X, y)
y_cat = []
converter = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1],
}
for label in y:
    y_cat.append(converter[label])


perceptron = Perceptron(4, 16, 3)


perceptron.train(X_train, y_train, learning_rate=0.01, epochs=40)

# predictions
y_pred = []
for element in X_test:
    y_pred.append(perceptron.predict(element))

# accuracy
T = 0
F = 0
for pred, g_truth in zip(y_pred, y_test):
    if np.argmax(pred, axis=0) == g_truth.index(1):
        T += 1
    else:
        F += 1

print(T / (T+F))
