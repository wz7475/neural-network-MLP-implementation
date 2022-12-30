import numpy as np


class Perceptron:
    def __init__(self, num_features, num_hidden_units, num_classes):
        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes
        self.W1 = np.zeros((num_hidden_units, num_features))
        self.W2 = np.zeros((num_classes, num_hidden_units))
        self.B1 = np.zeros(num_hidden_units)
        self.B2 = np.zeros(num_classes)

    def predict(self, X):
        # Calculate activations of hidden units
        Z1 = np.dot(self.W1, X) + self.B1
        self.Z1 = Z1
        # Apply activation function to hidden activations
        A1 = self._sigmoid(Z1)
        self.A1 = A1
        # Calculate activations of output units
        Z2 = np.dot(self.W2, A1) + self.B2
        self.Z2 = Z2
        # Apply activation function to output activations
        A2 = self._softmax(Z2)
        return A2

    def train(self, x, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for i in range(len(x)):
                Yh = self.predict(x[i])
                loss = self._categorical_crossentropy(y[i], Yh)
                dloss_Yh = self._dcategorical_crossentropy(y[i], Yh)
                # dloss_Z2 = dloss_Yh * self._dsoftmax(self.Z2)
                dloss_Z2 = np.dot(dloss_Yh, self._dsoftmax(self.Z2))
                dLoss_A1 = np.dot(self.W2.T, dloss_Z2)
                dloss_W2 = np.kron(dloss_Z2, self.A1).reshape(self.num_classes, self.num_hidden_units)
                # Calculate activations of hidden units
                hidden_activations = np.dot(self.W1, x[i]) + self.B1
                # Apply activation function to hidden activations
                hidden_output = self._sigmoid(hidden_activations)
                # Update weights and biases
                self.W2 += (learning_rate * loss * hidden_output[:, np.newaxis]).reshape(self.num_classes,
                                                                                         self.num_hidden_units)
                self.B2 += learning_rate * loss
                self.W1 += np.kron(learning_rate * hidden_output * (1 - hidden_output), x[i]).reshape(
                    self.num_hidden_units, self.num_features)
                self.B1 += learning_rate * hidden_output * (1 - hidden_output)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _softmax(self, x):
        return (np.exp(x) / np.exp(x).sum())

    def _dsoftmax(self, x):
        s = self._softmax(x)
        return np.diag(s) - np.outer(s, s)

    def _categorical_crossentropy(self, y_true, y_pred):
        # Ensure that y_pred is a valid probability distribution
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        # Calculate the negative log-likelihood of the true class
        return -np.sum(y_true * np.log(y_pred))

    def _dcategorical_crossentropy(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape


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
    y_cat.append(np.array(converter[label]))
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)


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

print(T / (T + F))
