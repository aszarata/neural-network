import numpy as np
import activation_function_base
from layer import Layer

class Network:

    def __init__(self):
        self.layers = []
        self.loss = self._MSE

    def _get_activation_function(self, type):
        if type == 'logistic':
            return (activation_function_base.logistic, activation_function_base.logistic_derivative)
        if type == 'identity':
            return (activation_function_base.identity, activation_function_base.identity_derivative)
        if type == 'relu':
            return (activation_function_base.relu, activation_function_base.relu_derivative)
        else:
            raise ValueError(f"Invalid activation function type: {type}")
        
    def _MSE(self, y_pred, y_true):
        loss = np.mean((y_pred.reshape(1, -1) - y_true.reshape(1, -1))**2)
        return loss
    

    def add_layer(self, in_features, out_features, activation_function='logistic'):
        activation_function, activation_derivative = self._get_activation_function(activation_function)
        self.layers.append(Layer(in_features, out_features, activation_function, activation_derivative))

    def fit(self, X, y, learning_rate=0.1, epochs=1000, verbose=-1):

        for epoch in range(epochs):
            for i in range(len(X)):
                inputs = X[i]
                for layer in self.layers:
                    inputs = layer.train_forward(inputs)
                
                outputs = y[i]
                output_layer = layer

                delta_tmp = output_layer.calculate_output_delta(outputs)
                output_layer.update_weights(lr=learning_rate)
                
                
                for layer in reversed(self.layers[:-1]):
                    delta_tmp = layer.calculate_delta(delta_tmp)
                    layer.update_weights(lr=learning_rate)

            if verbose!=-1 and epoch%10**verbose == 0:
                y_pred = self.predict(X)
                print(f"Epoch {epoch}: loss = {self._MSE(y_pred=y_pred, y_true=y)}")    
                
            
    
    def predict(self, X):
        y_pred = X

        for layer in self.layers:
            y_pred = layer.forward(y_pred)

        return y_pred

    
                



    
        
    