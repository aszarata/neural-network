import numpy as np
# from icecream import ic  # nice for debugging
import itertools


def hard_threshold(x):
    return np.where(x > 0, 1, 0)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

def back_prop_learning(X, y, network, activation_function=logistic, activation_derivative=logistic_derivative, learning_rate=0.1, epochs=10000):
    L = len(network)

    for _ in range(epochs):
        for i in range(len(X)):
            inputs = list(itertools.accumulate(
                network,
                np.dot,
                initial=X[i]
            ))  # functional operation instead of append

            net_output = activation_function(inputs[-1])  # single output evaluation

            output_derivative = activation_derivative(inputs[-1])
            residuum = (y[i] - net_output).reshape(-1, 1)  # reshape column vector to add 2-nd dimension instead of checking type
            
            deltas = output_derivative.T @ residuum
            
            for l in range(L - 2, 0, -1):
                delta = network[l + 1].T @ deltas[-1]
                deltas.append(activation_derivative(inputs[l]).T @ delta)  # accumulate is very complex here; append vs. insert...

            # deltas = deltas[::-1]
            
            # for l in range(L - 1):
            #     outer = np.outer(inputs[l], deltas[l])
            #     network[l] += learning_rate * outer

            for layer, input, delta in zip(network[:-1], inputs, reversed(deltas)):  # if you will
                layer += learning_rate * np.outer(input, delta)

    return network

X = np.array([
    [0, 0, 1, 0, 0], 
    [0, 1, 1, 0, 0], 
    [1, 0, 0, 1, 0], 
    [1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1], 
    [0, 1, 0, 1, 0]])
y = np.array([[0], [0], [0], [1], [1], [1], [0]])


layers = [5,10, 1]

weights = [np.random.uniform(low=-1, high=1, size=(layers[i], layers[i + 1])) for i in range(len(layers) - 1)]

learned_weights = back_prop_learning(X, y, weights, activation_function=logistic, epochs=1000)


print("weights:")
for layer, w in enumerate(learned_weights):
    print(f"Layer {layer} weights:\n{w}")

y_pred = X @ weights[0] @ weights[1]

print("y_pred: ", logistic(y_pred))
