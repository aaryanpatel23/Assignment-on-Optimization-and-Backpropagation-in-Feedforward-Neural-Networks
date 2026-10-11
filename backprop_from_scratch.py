# Backpropagation from scratch for a one-hidden-layer MLP
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[0.2,0.7],[0.8,0.3]])
y = np.array([[0.1],[0.6],[0.5],[0.9],[0.55],[0.6]])

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

W1 = np.random.randn(2,4) * 0.5
b1 = np.zeros((1,4))
W2 = np.random.randn(4,1) * 0.5
b2 = np.zeros((1,1))
lr, epochs = 0.5, 5000
losses = []

for _ in range(epochs):
    h = sigmoid(X @ W1 + b1)
    pred = sigmoid(h @ W2 + b2)
    error = pred - y
    losses.append(np.mean(error ** 2))
    d_out = (2 / y.size) * error * sigmoid_derivative(pred)
    dW2, db2 = h.T @ d_out, np.sum(d_out, axis=0, keepdims=True)
    d_hidden = (d_out @ W2.T) * sigmoid_derivative(h)
    dW1, db1 = X.T @ d_hidden, np.sum(d_hidden, axis=0, keepdims=True)
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print("Initial MSE:", round(losses[0], 6))
print("Final MSE:", round(losses[-1], 6))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Backpropagation Training Loss")
plt.grid(True)
plt.show()
