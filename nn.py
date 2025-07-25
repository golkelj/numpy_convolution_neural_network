import numpy as np
import pandas as pd
from scipy import signal
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0

df = pd.DataFrame(x_train_flat)
df['label'] = y_train

data = np.array(df)
np.random.shuffle(data)

m, n = data.shape
data_test = data[0:10000].T
Y_test = data_test[n-1].astype(int)
X_test = data_test[0:n-1].astype(np.float32)

data_train = data[10000:m].T
Y_train = data_train[n-1].astype(int)
X_train = data_train[0:n-1].astype(np.float32)


def init_params():
    w1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def ReLu(Z):
    return np.maximum(0, Z)

def derivative_ReLu(Z):
    return Z > 0

def Softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, w1, b1, w2, b2):
    z1 = np.dot(w1, X) + b1
    a1 = ReLu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = Softmax(z2)
    return a1, a2, z1, z2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backward_propagation(X, z1, a1, w2, a2, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = a2 - one_hot_Y
    dW2 = np.dot(dZ2, a1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(w2.T, dZ2) * derivative_ReLu(z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, learning_rate):
    w1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2

def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        a1, a2, z1, z2 = forward_propagation(X, w1, b1, w2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, z1, a1, w2, a2, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            one_hot_Y = one_hot(Y)
            loss = -np.mean(np.log(a2[Y, np.arange(Y.size)] + 1e-8))  # add epsilon for stability

            print(f"Iteration {i}: Loss = {loss:.4f}")
    return w1, b1, w2, b2

# training
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, iterations=2000, alpha=0.01)

# === Define prediction and accuracy functions ===
def predict(X, w1, b1, w2, b2):
    _, a2, _, _ = forward_propagation(X, w1, b1, w2, b2)
    return np.argmax(a2, axis=0)

def accuracy(X, Y, w1, b1, w2, b2):
    preds = predict(X, w1, b1, w2, b2)
    return np.mean(preds == Y) * 100

# === Test on training and dev sets ===
train_acc = accuracy(X_train, Y_train, w1, b1, w2, b2)
dev_acc = accuracy(X_dev, Y_dev, w1, b1, w2, b2)

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Dev Accuracy: {dev_acc:.2f}%")

# === (Optional) Test on official MNIST test set ===
x_test_flat = x_test.reshape(-1, 28*28).T.astype(np.float32) / 255.0
y_test = y_test.astype(int)

test_acc = accuracy(x_test_flat, y_test, w1, b1, w2, b2)
print(f"Test Accuracy: {test_acc:.2f}%")
