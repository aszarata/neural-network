import numpy as np
import pickle
import activation_function_base
from layer import Layer

class Network:
    """
    This class represents a neural network, allowing for the construction,
    training, evaluation, and saving/loading of the model.
    """

    def __init__(self):
        """
        Initializes the network with an empty list of layers and sets the 
        loss function to mean squared error (MSE). Also sets early stopping
        patience to None.
        """
        self.layers = []
        self.loss_fn = self._MSE
        self._early_stopping_patience = None

    def _MSE(self, y_pred, y_true):
        """
        Computes the Mean Squared Error (MSE) loss between predictions and true values.
        
        Args:
            y_pred: Predicted values.
            y_true: True values.
        
        Returns:
            Mean squared error loss.
        """
        loss = np.mean((y_pred.reshape(1, -1) - y_true.reshape(1, -1))**2)
        return loss
    
    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        
        Args:
            layer: An instance of the Layer class to be added to the network.
        """
        self.layers.append(layer)

    def fit(self, X, y, learning_rate=0.1, epochs=1000, verbose=-1, batch_size=None):
        """
        Trains the neural network using the provided training data.
        
        Args:
            X: Training input data.
            y: Training target data.
            learning_rate: Learning rate for weight updates.
            epochs: Number of training epochs.
            verbose: Verbosity level for logging progress.
            batch_size: Size of mini-batches for training.
        """
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

            if verbose != -1 and epoch % 10**verbose == 0:
                y_pred = self.predict(X)
                print(f"Epoch {epoch}: loss = {self.loss_fn(y_pred=y_pred, y_true=y)}")
            
            if self._early_stop(X, y):
                break

    def predict(self, X):
        """
        Makes predictions using the trained neural network.
        
        Args:
            X: Input data for making predictions.
        
        Returns:
            Predictions made by the network.
        """
        y_pred = X
        for layer in self.layers:
            y_pred = layer.forward(y_pred)
        return y_pred

    def _generate_mini_batch(self, X, y, size):
        """
        Generates a mini-batch from the training data.
        
        Args:
            X: Training input data.
            y: Training target data.
            size: Size of the mini-batch.
        
        Returns:
            A mini-batch of input and target data.
        """
        if size is None:
            return X, y

        start_idx = np.random.randint(0, len(X) - size + 1)
        mini_batch_X = X[start_idx:start_idx + size]
        mini_batch_y = y[start_idx:start_idx + size]
        return mini_batch_X, mini_batch_y

    def early_stopping(self, early_stopping_patience):
        """
        Enables early stopping to prevent overfitting during training.
        
        Args:
            early_stopping_patience: Number of epochs with no improvement
            after which training will be stopped.
        """
        self._early_stopping_patience = early_stopping_patience
        self.__steps_without_improvement = 0
        self.__best_val_loss = np.inf

    def _early_stop(self, X, y):
        """
        Checks if early stopping criteria are met.
        
        Args:
            X: Validation input data.
            y: Validation target data.
        
        Returns:
            Boolean indicating whether to stop training.
        """
        if self._early_stopping_patience is None:
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

    def save(self, filename):
        """
        Saves the trained model to a file.
        
        Args:
            filename: Path to the file where the model will be saved.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """
        Loads a trained model from a file.
        
        Args:
            filename: Path to the file from which the model will be loaded.
        
        Returns:
            The loaded model.
        """
        with open(filename, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object
