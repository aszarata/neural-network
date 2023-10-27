from multilayer_network_alternative import back_prop_learning, logistic
import numpy as np


X = np.array([
    [0, 0, 1, 0, 0], 
    [0, 1, 1, 0, 0], 
    [1, 0, 0, 1, 0], 
    [1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1], 
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 1, 0]])
y = np.array([[0], [0], [0], [1], [1], [1], [0], [0], [1], [1], [1], [1], [0]])


layers = [5, 1]

weights = [np.random.uniform(low=-1, high=1, size=(layers[i], layers[i + 1])) for i in range(len(layers) - 1)]

learned_weights = back_prop_learning(X, y, weights, activation_function=logistic, learning_rate= 0.1, epochs=10000)



X_test = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 1, 0]
])

y_test = [
    [0],
    [1],
    [1],
    [1],
    [1],
    [0]
]

y_pred = X_test
for weight in weights:
    y_pred = y_pred @ weight



print(np.round(logistic(y_pred)), y_test)
