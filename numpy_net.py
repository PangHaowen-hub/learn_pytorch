import numpy as np

x = np.random.randn(64, 1000)
z = np.random.randn(64, 10)

w1 = np.random.randn(1000, 100)
w2 = np.random.randn(100, 10)

learning_rate = 1e-6

for t in range(500):
    y = x.dot(w1)
    y_relu = np.maximum(y, 0)
    z_pred = y_relu.dot(w2)

    loss = np.square(z_pred - z).sum()
    print(t, loss)

    grad_z_pred = 2.0 * (z_pred - z)
    grad_w2 = y_relu.T.dot(grad_z_pred)
    grad_y_relu = grad_z_pred.dot(w2.T)
    grad_y = grad_y_relu.copy()
    grad_y[y < 0] = 0
    grad_w1 = x.T.dot(grad_y)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
y = x.dot(w1)
y_relu = np.maximum(y, 0)
z_pred = y_relu.dot(w2)
print(z_pred - z)
