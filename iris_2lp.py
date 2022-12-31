import numpy as np


class Perceptron:
    def __init__(self, num_features, num_hidden_units_1, num_hidden_units_2, num_classes):
        self.num_features = num_features
        self.num_hidden_units_1 = num_hidden_units_1
        self.num_hidden_units_2 = num_hidden_units_2
        self.num_classes = num_classes
        self.num_layers = 3
        self.W = []
        self.B = []
        self.Z = []
        self.A = []
        self._init_params()
        self.losses = []

    def _init_params(self):
        self.W.append(np.random.rand(self.num_hidden_units_1, self.num_features))
        self.W.append(np.random.rand(self.num_hidden_units_2, self.num_hidden_units_1))
        self.W.append(np.random.rand(self.num_classes, self.num_hidden_units_2))
        self.B.append(np.random.rand(self.num_hidden_units_1))
        self.B.append(np.random.rand(self.num_hidden_units_2))
        self.B.append(np.random.rand(self.num_classes))
        for _ in range(self.num_layers):
            # create list placeholders for Z and A
            self.Z.append(None)
            self.A.append(None)

    def predict_sample(self, X):
        Z1 = np.dot(self.W[0], X) + self.B[0]
        self.Z[0] = Z1
        A1 = self._sigmoid(Z1)
        self.A[0] = A1

        Z2 = np.dot(self.W[1], A1) + self.B[1]
        self.Z[1] = Z2
        A2 = self._sigmoid(Z2)
        self.A[1] = A2

        Z3 = np.dot(self.W[2], A2) + self.B[2]
        self.Z[2] = Z3
        A3 = self._softmax(Z3)
        return A3

    def predict_batch(self, X):
        predictions = []
        for element in X:
            predictions.append(self.predict_sample(element))
        return np.array(predictions)

    def train(self, x, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            total_losses = 0
            for i in range(len(x)):
                x_i = np.array(x[i])
                Yh = self.predict_sample(x[i])
                loss = self._categorical_crossentropy(y[i], Yh)
                total_losses += loss

                dloss_Yh = self._dcategorical_crossentropy(y[i], Yh)
                dloss_A2 = dloss_Yh
                dloss_Z2 = np.dot(dloss_A2, self._dsoftmax(self.Z[2]))
                dLoss_A1 = np.dot(self.W[2].T, dloss_Z2)
                dloss_W2 = np.kron(dloss_Z2, self.A[1]).reshape(self.num_classes, self.num_hidden_units_2)
                dloss_B2 = dloss_Z2

                dloss_Z1 = dLoss_A1 * self._dsigmoid(self.Z[1])
                dLoss_A0 = np.dot(self.W[1].T, dloss_Z1)
                dloss_W1 = np.kron(dloss_Z1, self.A[0]).reshape(self.num_hidden_units_2, self.num_hidden_units_1)
                dloss_B1 = dloss_Z1

                dloss_Z0 = dLoss_A0 * self._dsigmoid(self.Z[0])
                dloss_W0 = np.kron(dloss_Z0, x_i).reshape(self.num_hidden_units_1, self.num_features)
                dloss_B0 = dloss_Z0


                self.W[2] -= learning_rate * dloss_W2
                self.B[2] -= learning_rate * dloss_B2

                self.W[1] -= learning_rate * dloss_W1
                self.B[1] -= learning_rate * dloss_B1

                self.W[0] -= learning_rate * dloss_W0
                self.B[0] -= learning_rate * dloss_B0
            self.losses.append(total_losses / len(x))
            if epoch % 50 == 0:
                train_cross_entripy = self._categorical_crossentropy(y, self.predict_batch(x)) / len(y)
                print(f"epoch: {epoch}; train loss: {train_cross_entripy}")
                val_cross_entripy = self._categorical_crossentropy(y_test, self.predict_batch(X_test)) / len(y_test)
                print(f"validation loss: {val_cross_entripy}")
                print()

    def score(self, X, y):
        predictions = self.predict_batch(X)
        return np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)

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

# X, y = datasets.load_iris(return_X_y=True)
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

# perceptron = Perceptron(4, 16, 3)
perceptron = Perceptron(4, 16, 8, 3)

perceptron.train(X_train, y_train, learning_rate=0.01, epochs=800)

# predictions


print(perceptron.predict_batch(X_test))
print(perceptron.losses)

print(perceptron.score(X_test, y_test))
print()
