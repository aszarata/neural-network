import numpy as np
import itertools
import activation_function_base

class Network:

    def __init__(self):
        self.weights = []
        self.layers = []

    def fit(self, X, y, learning_rate=0.1, epochs=100, activation_function = 'logistic'):
        
        
        activation_function, activation_derivative = self._get_activation_function(activation_function)
        

        self.layers = [X.shape[1]] + self.layers + [y.shape[1] if len(y.shape) > 1 else 1]
        
        
        self._init_weights()
        L = len(self.weights)
        

        for epoch in range(epochs):
            for i in range(len(X)):
                inputs = list(itertools.accumulate(
                    self.weights,
                    np.dot,
                    initial=X[i]
                ))  # functional operation instead of append

                net_output = activation_function(inputs[-1])  # single output evaluation

                output_derivative = activation_derivative(inputs[-1])
                residuum = (y[i] - net_output).reshape(-1, 1)  # reshape column vector to add 2-nd dimension instead of checking type
                
                deltas = output_derivative.T @ residuum
                
                for l in range(L - 2, 0, -1):
                    delta = self.weights[l + 1].T @ deltas[-1]
                    deltas.append(activation_derivative(inputs[l]).T @ delta)  # accumulate is very complex here; append vs. insert...

                for layer, input, delta in zip(self.weights[:-1], inputs, reversed(deltas)):  # if you will
                    layer += learning_rate * np.outer(input, delta)

    def add_layer(self, size):
        self.layers.append(size)

    def predict(self, X):
        y_pred = X
        for weight in self.weights:
            y_pred = y_pred @ weight
        
        return y_pred

    def _init_weights(self):
        self.weights = [np.random.uniform(low=-1, high=1, size=(self.layers[i], self.layers[i + 1])) for i in range(len(self.layers) - 1)]

    def _get_activation_function(self, type):
        if type == 'logistic':
            return (activation_function_base.logistic, activation_function_base.logistic_derivative)
        if type == 'identity':
            return (activation_function_base.identity, activation_function_base.identity_derivative)
        else:
            raise ValueError(f"Invalid activation function type: {type}")
