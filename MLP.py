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
        self.W.append(np.random.rand(self.num_classes, self.hidden_layers[last_hidden])) # hidden - output

        for i in range(first_hidden, last_hidden+1):
            self.B.append(np.random.rand(self.hidden_layers[i]))
        # self.B.append(np.random.rand(self.hidden_layers[0])) # first hidden
        # self.B.append(np.random.rand(self.hidden_layers[1]))
        # self.B.append(np.random.rand(self.hidden_layers[2])) # last hidden
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

                last_hidden = self.num_layers - 2
                first_hidden = 1

                # output layer
                dloss_Yh = self.loss_function(y_train[i], Yh, der=True)
                dloss_A3 = dloss_Yh
                dout = self.activations[-1](self.Z[-1], der=True)

                if(dout.size == 9):
                    dloss_Z = np.dot(dloss_A3, dout)
                else:
                    dloss_Z = dloss_A3 * dout

                dloss_A = np.dot(self.W[-1].T, dloss_Z)
                dloss_W3 = np.kron(dloss_Z, self.A[-1]).reshape(self.num_classes, self.hidden_layers[last_hidden])
                dloss_B3 = dloss_Z


                hd_dloss_Z = []
                hd_dloss_A = []
                hd_dloss_W = []
                hd_dloss_B = []

                for i_hd_lay in reversed(range(first_hidden, last_hidden+1)):
                    dloss_Z = dloss_A * self.activations[i_hd_lay](self.Z[i_hd_lay], der=True)
                    hd_dloss_Z.insert(0, dloss_Z)
                    dloss_A = np.dot(self.W[i_hd_lay].T, dloss_Z)
                    hd_dloss_A.insert(0, dloss_A)
                    dloss_W1 = np.kron(dloss_Z, self.A[i_hd_lay - 1]).reshape(self.hidden_layers[i_hd_lay],
                                                                                   self.hidden_layers[i_hd_lay - 1])
                    hd_dloss_W.insert(0, dloss_W1)
                    dloss_B1 = dloss_Z
                    hd_dloss_B.insert(0, dloss_B1)

                # input layer
                dloss_Z0 = dloss_A * self.activations[0](self.Z[0], der=True)
                dloss_W0 = np.kron(dloss_Z0, x_i).reshape(self.hidden_layers[0], self.num_features)
                dloss_B0 = dloss_Z0

                # output layer
                self.W[-1] -= learning_rate * dloss_W3
                self.B[-1] -= learning_rate * dloss_B3

                # hidden layers
                for i in range(first_hidden, last_hidden):
                    self.W[i+1] -= learning_rate * hd_dloss_W[i]
                    self.B[i+1] -= learning_rate * hd_dloss_B[i]
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
hidden_layers = [8, 7, 6, 5]
hidden_layers = [16]
activations = [sigmoid] * len(hidden_layers) + [softmax]
perceptron = Perceptron(13, hidden_layers, 3, activations, loss_function=ccategorical_crossentropy)

perceptron.train(X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=800)

# predictions


print(perceptron.predict_batch(X_test))

print(perceptron.score(X_test, y_test))
print()
