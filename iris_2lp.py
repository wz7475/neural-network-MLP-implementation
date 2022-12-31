import numpy as np


class Perceptron:
    def __init__(self, num_features, hidden_layers, num_classes, activations, loss_function):
        self.num_features = num_features
        self.activations = activations
        self.loss_function = loss_function
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.num_layers = len(hidden_layers) + 1
        self.W = []
        self.B = []
        self.Z = []
        self.A = []
        self.history = None
        self._init_params()

    def _init_params(self):
        self.W.append(np.random.rand(self.hidden_layers[0], self.num_features)) # input - hidden
        first_hidden = 0
        last_hidden = self.num_layers - 2
        for i in range(first_hidden, last_hidden):
            self.W.append(np.random.rand(self.hidden_layers[i+1], self.hidden_layers[i]))
        # self.W.append(np.random.rand(self.hidden_layers[1], self.hidden_layers[0]))
        # self.W.append(np.random.rand(self.hidden_layers[2], self.hidden_layers[1]))
        self.W.append(np.random.rand(self.num_classes, self.hidden_layers[2])) # hidden - output
        self.B.append(np.random.rand(self.hidden_layers[0])) # input
        self.B.append(np.random.rand(self.hidden_layers[1]))
        self.B.append(np.random.rand(self.hidden_layers[2]))
        self.B.append(np.random.rand(self.num_classes)) # output
        for _ in range(self.num_layers):
            # create list placeholders for Z and A
            self.Z.append(None)
        for _ in range(self.num_layers - 1):
            self.A.append(None)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def predict_sample(self, X):
        Z0 = np.dot(self.W[0], X) + self.B[0]
        self.Z[0] = Z0
        A0 = self.activations[0](Z0)
        self.A[0] = A0

        # Z1 = np.dot(self.W[1], A0) + self.B[1]
        # self.Z[1] = Z1
        # A1 = self.activations[1](Z1)
        # self.A[1] = A1
        #
        # Z2 = np.dot(self.W[2], A1) + self.B[2]
        # self.Z[2] = Z2
        # A2 = self.activations[2](Z2)
        # self.A[2] = A2

        for i in range(1, self.num_layers-1):
            Z = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            self.Z[i] = Z
            A = self.activations[i](Z)
            self.A[i] = A

        Z_last = np.dot(self.W[-1], self.A[-1]) + self.B[-1]
        self.Z[-1] = Z_last
        A_last = self.activations[-1](Z_last)
        return A_last

    def predict_batch(self, X):
        predictions = []
        for element in X:
            predictions.append(self.predict_sample(element))
        return np.array(predictions)

    def train(self, x_train, y_train, x_test, y_test, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            total_losses = 0
            for i in range(len(x_train)):
                x_i = np.array(x_train[i])
                Yh = self.predict_sample(x_train[i])
                loss = self.loss_function(y_train[i], Yh)
                total_losses += loss

                # output layer
                dloss_Yh = self.loss_function(y_train[i], Yh, der=True)
                dloss_A3 = dloss_Yh
                dloss_Z3 = np.dot(dloss_A3, self.activations[-1](self.Z[-1], der=True))
                dloss_A2 = np.dot(self.W[-1].T, dloss_Z3)
                dloss_W3 = np.kron(dloss_Z3, self.A[-1]).reshape(self.num_classes, self.hidden_layers[2])
                dloss_B3 = dloss_Z3

                # last hidden layer
                last_hidden = self.num_layers - 2
                dloss_Z2 = dloss_A2 * self.activations[last_hidden](self.Z[last_hidden], der=True)
                dLoss_A1 = np.dot(self.W[last_hidden].T, dloss_Z2)
                dloss_W2 = np.kron(dloss_Z2, self.A[last_hidden - 1]).reshape(self.hidden_layers[last_hidden],
                                                                              self.hidden_layers[last_hidden - 1])
                dloss_B2 = dloss_Z2

                # first hidden layer
                first_hidden = 1
                dloss_Z1 = dLoss_A1 * self.activations[first_hidden](self.Z[first_hidden], der=True)
                dLoss_A0 = np.dot(self.W[first_hidden].T, dloss_Z1)
                dloss_W1 = np.kron(dloss_Z1, self.A[first_hidden - 1]).reshape(self.hidden_layers[first_hidden],
                                                                               self.hidden_layers[first_hidden - 1])
                dloss_B1 = dloss_Z1

                # input layer
                dloss_Z0 = dLoss_A0 * self.activations[0](self.Z[0], der=True)
                dloss_W0 = np.kron(dloss_Z0, x_i).reshape(self.hidden_layers[0], self.num_features)
                dloss_B0 = dloss_Z0

                # output layer
                self.W[-1] -= learning_rate * dloss_W3
                self.B[-1] -= learning_rate * dloss_B3

                # hidden layers
                self.W[2] -= learning_rate * dloss_W2
                self.B[2] -= learning_rate * dloss_B2

                self.W[1] -= learning_rate * dloss_W1
                self.B[1] -= learning_rate * dloss_B1

                # input layer
                self.W[0] -= learning_rate * dloss_W0
                self.B[0] -= learning_rate * dloss_B0
            # evaluate the model
            self.history["train_acc"].append(self.score(x_train, y_train))
            self.history["val_acc"].append(self.score(x_test, y_test))
            self.history["train_loss"].append(
                self.loss_function(y_train, self.predict_batch(x_train)) / len(y_train))
            self.history["val_loss"].append(
                self.loss_function(y_test, self.predict_batch(x_test)) / len(y_test))
            if epoch % 50 == 0:
                print(
                    f"epoch: {epoch}; train loss: {self.history['train_loss'][-1]}; val loss: {self.history['val_loss'][-1]}")
                print(f"train acc: {self.history['train_acc'][-1]}; val acc: {self.history['val_acc'][-1]}\n")

    def score(self, X, y):
        predictions = self.predict_batch(X)
        return np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)


def ccategorical_crossentropy(y_true, y_pred, der=False):
    if der:
        return (y_pred - y_true) / y_true.shape
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.sum(y_true * np.log(y_pred))


def sigmoid(x, der=False):
    if der:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def softmax(x, der=False):
    if der:
        s = softmax(x)
        return np.diag(s) - np.outer(s, s)
    return (np.exp(x) / np.exp(x).sum())


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

# X, y = datasets.load_iris(return_X_y=True)
X, y = datasets.load_wine(return_X_y=True)
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

# perceptron = Perceptron(4, 16, 3)
activations = [sigmoid, sigmoid, sigmoid, softmax]
perceptron = Perceptron(13, [8, 8, 4], 3, activations, loss_function=ccategorical_crossentropy)

perceptron.train(X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=800)

# predictions


print(perceptron.predict_batch(X_test))

print(perceptron.score(X_test, y_test))
print()
