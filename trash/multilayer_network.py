import numpy as np


def hard_threshold(x):
    return np.where(x > 0, 1, 0)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def derivative(f, x, h = 10e-5):
    return (f(x+h) - f(x-h)) / 2

def back_prop_learning(X, y, network, activation_function, learning_rate=0.1, epochs=10000):
    L = len(network)

    for epoch in range(epochs):
        for i in range(len(X)):

            #forward
            inputs = [X[i]]
            for l in range(L):
                net_input = np.dot(inputs[l], network[l])
                net_output = activation_function(net_input)
                inputs.append(net_input)

            deltas = derivative(activation_function, net_input).T @ (y[i] - net_output)
            print(deltas)
            if type(deltas) != np.ndarray: deltas = np.array([deltas])
            for l in range(L - 2, 0, -1):
                delta = network[l+1].T @ deltas[0]
                deltas.insert(0, derivative(activation_function, inputs[l]).T @ delta)


            #updating weights
            for l in range(1, L):
                network[l] += learning_rate * np.outer(inputs[l - 1], deltas[l - 1])

    return network

X = np.array([
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]])
y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


layers = [2, 3, 4]

weights = [np.random.uniform(low=-1, high=1, size=(layers[i], layers[i + 1])) for i in range(len(layers) - 1)]

learned_weights = back_prop_learning(X, y, weights, activation_function=logistic, epochs=10)


print("weights:")
for layer, w in enumerate(learned_weights):
    print(f"Layer {layer} weights:\n{w}")

y_pred = X @ weights[0] @ weights[1]

print("X_pred: ", logistic(y_pred))