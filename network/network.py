import numpy as np
import pickle
import activation_function_base
from layer import Layer

class Network:

    # Init
    def __init__(self):
        self.layers = []
        self.loss_fn = self._MSE

        self._early_stopping_patience = None

    def _get_activation_function(self, type):
        if type == 'logistic':
            return (activation_function_base.logistic, activation_function_base.logistic_derivative)
        if type == 'identity':
            return (activation_function_base.identity, activation_function_base.identity_derivative)
        if type == 'relu':
            return (activation_function_base.relu, activation_function_base.relu_derivative)
        if type == 'softmax':
            return (activation_function_base.softmax, activation_function_base.softmax_derivative)
        else:
            raise ValueError(f"Invalid activation function type: {type}")
        
    def _MSE(self, y_pred, y_true):
        loss = np.mean((y_pred.reshape(1, -1) - y_true.reshape(1, -1))**2)
        return loss
    
    # Network building
    def add_layer(self, 
                  in_features, 
                  out_features, 
                  activation_function='logistic', 
                  dropout_prob=None,
                  batch_norm_1d_size=None):
        
        activation_function, activation_derivative = self._get_activation_function(activation_function)
        
        self.layers.append(
            Layer(in_features, 
                  out_features, 
                  activation_function, 
                  activation_derivative, 
                  dropout_prob,
                  batch_norm_1d_size)
            )

    # Training
    def fit(self, X, y, learning_rate=0.1, epochs=1000, verbose=-1, batch_size=None):

        for epoch in range(epochs):
            X_batch, y_batch = self._generate_mini_batch(X=X, y=y, size=batch_size)
            for i in range(len(X_batch)):
                inputs = X_batch[i]
                for layer in self.layers:
                    inputs = layer.train_forward(inputs)
                
                outputs = y_batch[i]
                output_layer = layer

                delta_tmp = output_layer.calculate_output_delta(outputs)
                output_layer.update_weights(lr=learning_rate)
                
                
                for layer in reversed(self.layers[:-1]):
                    delta_tmp = layer.calculate_delta(delta_tmp)
                    layer.update_weights(lr=learning_rate)

            if verbose!=-1 and epoch%10**verbose == 0:
                y_pred = self.predict(X)
                print(f"Epoch {epoch}: loss = {self.loss_fn(y_pred=y_pred, y_true=y)}")
            
            if self._early_stop(X, y):
                break
                
            
    # Evaluations
    def predict(self, X):
        y_pred = X

        for layer in self.layers:
            y_pred = layer.forward(y_pred)

        return y_pred
    
    # Mini batches
    def _generate_mini_batch(self, X, y, size):
        if size==None:
            return X, y

        start_idx = np.random.randint(0, len(X) - size + 1)

        mini_batch_X = X[start_idx:start_idx + size]
        mini_batch_y = y[start_idx:start_idx + size]

        return mini_batch_X, mini_batch_y
    
    # Early stopping
    def early_stopping(self, early_stopping_patience):
        self._early_stopping_patience = early_stopping_patience
        self.__steps_without_improvement = 0
        self.__best_val_loss = np.inf

    def _early_stop(self, X, y):
        if self._early_stopping_patience == None:
            return False
        
        y_pred = self.predict(X)
        loss = self.loss_fn(y_pred=y_pred, y_true=y)
        if loss < self.__best_val_loss:
            self.__best_val_loss = loss
            self.__steps_without_improvement = 0
        
        else:
            self.__steps_without_improvement += 1

        if self.__steps_without_improvement == self._early_stopping_patience:
            return True
        
        return False
    
    # Save and load model
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object

    
                



    
        
    