import numpy as np  # linear algebra

np.random.seed(10)
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Required magic to display matplotlib plots in notebooks

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

data = pd.read_csv('../data/train.csv')

data.head(4)
# %% md
# The Data

dict_live = {
    0: 'Perished',
    1: 'Survived'
}
dict_sex = {
    'male': 0,
    'female': 1
}
data['Bsex'] = data['Sex'].apply(lambda x: dict_sex[x])
# manual encoding to numbers

# features - matrix with 2 features columns
# labels - matrix with one column - Y - survived or no
features = data[['Pclass', 'Bsex']].to_numpy()
labels = data['Survived'].to_numpy()


# %%
# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    import numpy as np

    if (der == True):  # derivative of the sigmoid
        f = x / (1 - x)
    else:  # sigmoid
        f = 1 / (1 + np.exp(-x))

    return f


# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=False):
    import numpy as np

    if (der == True):
        if x > 0:
            f = 1
        else:
            f = 0
    else:
        if x > 0:
            f = x
        else:
            f = 0
    return f


# one perceptron, just to show weights multiplication
def perceptron(X, act='Sigmoid'):
    shapes = X.shape
    n = shapes[0] + shapes[1]
    w = 2 * np.random.random(shapes) - 0.5  # We want w to be between -1 and 1
    b = np.random.random(1)

    f = b[0]  # init with output bias

    # multiply X(input) * w(weights) and sum elements
    f += np.sum(np.multiply(X, w)) / n
    # for i in range(0, X.shape[0]-1) : # run over column elements
    #     for j in range(0, X.shape[1]-1) : # run over rows elements
    #         f += w[i, j]*X[i,j]/n

    if act == 'Sigmoid':
        output = sigmoid_act(f)
    else:
        output = ReLU_act(f)

    return output


print('Output with sigmoid activator: ', perceptron(features))
print('Output with ReLU activator: ', perceptron(features))
# %%
import numpy as np


# more fancy declarations
def sigmoid_act(x, der=False):
    import numpy as np

    if (der == True):  # derivative of the sigmoid
        f = 1 / (1 + np.exp(- x)) * (1 - 1 / (1 + np.exp(- x)))
    else:  # sigmoid
        f = 1 / (1 + np.exp(- x))

    return f


# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=False):
    import numpy as np

    if (der == True):  # the derivative of the ReLU is the Heaviside Theta
        f = np.heaviside(x, 1)
    else:
        f = np.maximum(x, 0)

    return f


# %%
# split into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)

print('Training records:', Y_train.size)
print('Test records:', Y_test.size)
# %%
num_neurons_lay_1 = 4
num_neurons_layer_2 = 4
lr = 1 / 623

w1 = 2 * np.random.rand(num_neurons_lay_1, X_train.shape[1]) - 0.5
b1 = np.random.rand(num_neurons_lay_1)

w2 = 2 * np.random.rand(num_neurons_layer_2, num_neurons_lay_1) - 0.5
b2 = np.random.rand(num_neurons_layer_2)

wOut = 2 * np.random.rand(num_neurons_layer_2) - 0.5
bOut = np.random.rand(1)

losses = []
y_pred = []
# %%
for I in range(0, X_train.shape[0]):
    # x - training sample
    x = X_train[I]

    #  Feed forward
    z1 = ReLU_act(np.dot(w1, x) + b1)  # output layer 1
    # w1(4,2) x(2,) => z1(4,) vector 4 dot products 2x2
    z2 = ReLU_act(np.dot(w2, z1) + b2)  # output layer 2
    y = sigmoid_act(np.dot(wOut, z2) + bOut)  # Output of the Output layer

    # 2.2: Compute the output layer's error
    delta_Out = (y - Y_train[I]) * sigmoid_act(y, der=True)

    # 2.3: Backpropagate
    delta_2 = delta_Out * wOut * ReLU_act(z2, der=True)  # Second Layer Error
    delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True)  # First Layer Error

    # 3: Gradient descent
    wOut = wOut - lr * delta_Out * z2  # Outer Layer
    bOut = bOut - lr * delta_Out

    w2 = w2 - lr * np.kron(delta_2, z1).reshape(num_neurons_layer_2, num_neurons_lay_1)  # Hidden Layer 2
    b2 = b2 - lr * delta_2

    w1 = w1 - lr * np.kron(delta_1, x).reshape(num_neurons_lay_1, x.shape[0])  # Hidden Layer 1
    b1 = b1 - lr * delta_1

    # 4. Computation of the loss function
    losses.append((1 / 2) * (y - Y_train[I]) ** 2)
    y_pred.append(y[0])

# Plotting the Cost function for each training data
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(0, X_train.shape[0]), losses, alpha=0.3, s=4, label='mu')
plt.title('Loss for each training data point', fontsize=20)
plt.xlabel('Training data', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.show()

# Plotting the average cost function over 10 training data
pino = []
for i in range(0, 9):
    pippo = 0
    for m in range(0, 59):
        pippo += y_pred[60 * i + m] / 60
    pino.append(pippo)

plt.figure(figsize=(10, 6))
plt.scatter(np.arange(0, 9), pino, alpha=1, s=10, label='error')
plt.title('Averege Loss by epoch', fontsize=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.show()