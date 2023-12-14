import numpy as np
import itertools
import activation_function_base

class Network:

    def __init__(self):
        self.weights = []
        self.layers = []
        self.activation_function, self.activation_derivative = self._get_activation_function('relu')

    def fit(self, X, y, learning_rate=0.1, epochs=10000):
    

        self._init_weights()
        L = len(self.weights)

        y_pred = np.zeros(len(y))

        for epoch in range(epochs):
            for i in range(len(X)):
                inputs = [X[i]]
                for layer_weights in self.weights:
                    inputs.append( self.activation_function(np.dot(np.append(inputs[-1], -1), layer_weights)) )

                net_output = self.activation_function(inputs[-1]) 
                y_pred[i] = net_output

                output_derivative = self.activation_derivative(inputs[-1])
                residuum = (y[i] - net_output).reshape(-1, 1) 
                
                deltas = output_derivative.T @ residuum
                
                for l in range(L - 2, 0, -1):
                    delta = self.weights[l + 1].T @ deltas[-1]
                    deltas.append(self.activation_derivative(inputs[l]).T @ delta)

                for w, input, delta in zip(self.weights[:-1], inputs, reversed(deltas)): 
                    w += learning_rate * np.outer(np.append(input, 1), delta)
            
            print(f"Epoch {epoch}: loss = {self.__MSE(y_pred, y)}")


    def add_layer(self, size):
        self.layers.append(size)

    def set_activation_function(self, activation_function):
        self.activation_function, self.activation_derivative = self._get_activation_function(activation_function)

    def predict(self, X):
        y_pred = X
        for weight in self.weights:
            y_pred = self.activation_function(np.append(y_pred, -1) @ weight)
        
        return y_pred

    def _init_weights(self):
        self.weights = [np.random.uniform(low=-1, high=1, size=(self.layers[i] + 1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

    def _get_activation_function(self, type):
        if type == 'logistic':
            return (activation_function_base.logistic, activation_function_base.logistic_derivative)
        if type == 'identity':
            return (activation_function_base.identity, activation_function_base.identity_derivative)
        if type == 'relu':
            return (activation_function_base.relu, activation_function_base.relu_derivative)
        else:
            raise ValueError(f"Invalid activation function type: {type}")
        
    def __MSE(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

        


X_train = np.array([
    [0, 1], 
    [1, 1], 
    [1, 2], 
    [4, 0],
    [13, 2],
    [12, 5], 
    [3, 6]])
y_train = np.array([[1], [2], [3], [4], [15], [17], [9]])

model = Network()
model.set_activation_function('relu')
model.add_layer(X_train.shape[1])
model.add_layer(5)
model.add_layer(y_train.shape[1])
model.fit(X_train, y_train, learning_rate=0.3, epochs=100)

print(model.predict(np.array([[6, 6]])))