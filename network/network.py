import numpy as np
import itertools
import activation_function_base

class Network:

    def __init__(self):
        self.weights = []
        self.layers = []

    def fit(self, X, y, learning_rate=0.1, epochs=100, activation_function = 'relu'):
        
        
        activation_function, activation_derivative = self._get_activation_function(activation_function)
        
        
        self._init_weights()
        L = len(self.layers)
        

        for epoch in range(epochs):
            for i in range(len(X)):
                inputs = list(itertools.accumulate(
                    self.weights,
                    np.dot,
                    initial=X[i]
                ))  

                net_output = activation_function(inputs[-1]) 

                output_derivative = activation_derivative(inputs[-1])
                residuum = (y[i] - net_output) 
                
                deltas = [np.atleast_2d(output_derivative * residuum)]
                
                for l in range(L - 2, -1, -1):
                    delta = deltas[-1] @ self.weights[l].T 
                    deltas.append(activation_derivative(inputs[l]) * delta)  

                for layer, input, delta in zip(self.weights[:-1], inputs[1:], reversed(deltas)): 
                    layer += learning_rate * np.outer(delta, activation_function(input))

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
        if type == 'relu':
            return (activation_function_base.relu, activation_function_base.relu_derivative)
        else:
            raise ValueError(f"Invalid activation function type: {type}")


# X_train = np.array([
#     [0, 1], 
#     [1, 1], 
#     [1, 2], 
#     [4, 0],
#     [13, 2],
#     [12, 5], 
#     [3, 6]])
# y_train = np.array([[1], [2], [3], [4], [15], [17], [9]])

# model = Network()
# model.add_layer(X_train.shape[1])
# model.add_layer(4)
# model.add_layer(y_train.shape[1])
# model.fit(X_train, y_train, activation_function='identity', learning_rate=0.3, epochs=10)

# print(model.predict(np.array([[3, 6]])))