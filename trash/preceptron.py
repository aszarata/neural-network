import numpy as np

def hard_treshold(x):
    return np.where(x > 0, 1, 0)

def train(X, y, learning_rate = 0.5, epochs = 400, activation_function = hard_treshold):
    
    shape = X.shape[1], y.shape[1]
    w = np.random.uniform(low=-1, high=1, size=shape)


    for epoch in range(epochs):
        for i in range(len(X)):
            net_input = X[i] @ w
            net_output = np.transpose(activation_function(net_input))
            error = np.outer((y[i] - net_output), X[i])
            w += np.transpose(learning_rate * error)

    return w





X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0, 0],
    [0, 1],
    [0, 1],
    [1, 1]
])


training_weights = train(X, y)

y_pred = X @ training_weights

print(y_pred)