import numpy as np
import activation_function_base

class Layer:
    """
    This class represents a single layer in a neural network, supporting
    forward and backward propagation, weight updates, dropout, and batch normalization.
    """

    def __init__(self, in_features, out_features, activation_function='logistic', dropout_prob=None, batch_norm_1d_size=None):
        """
        Initializes the layer with the specified number of input and output features,
        activation function, dropout probability, and batch normalization size.

        Args:
            in_features: Number of input features to the layer.
            out_features: Number of output features from the layer.
            activation_function: Activation function to be used (default is 'logistic').
            dropout_prob: Probability of dropping out neurons during training (default is None).
            batch_norm_1d_size: Size for batch normalization (default is None).
        """
        self.weights = np.random.uniform(low=-1, high=1, size=(in_features + 1, out_features))
        self.activation_function, self.activation_derivative = activation_function_base.get_activation_function(activation_function)
        self.input_links = None
        self.input = None
        self.output = None
        self.delta = None
        self.dropout_prob = dropout_prob
        self.batch_norm_1d_size = batch_norm_1d_size
        self.bias = 1

    def train_forward(self, input_links):
        """
        Performs the forward pass during training with dropout and batch normalization.

        Args:
            input_links: Input data to the layer.
        
        Returns:
            Output of the layer after applying the activation function.
        """
        self.input_links = np.append(input_links, self.bias)  # Add bias term
        self.input = self.input_links @ self.weights
        self._apply_dropout_mask()  # Apply dropout
        self._apply_batch_norm()  # Apply batch normalization
        self.output = self.activation_function(self.input)
        return self.output

    def forward(self, input_links):
        """
        Performs the forward pass without dropout and batch normalization (for inference).

        Args:
            input_links: Input data to the layer.
        
        Returns:
            Output of the layer after applying the activation function.
        """
        bias_column = np.full((input_links.shape[0], 1), self.bias)
        input_links = np.hstack((input_links, bias_column))
        return self.activation_function(input_links @ self.weights)

    def calculate_output_delta(self, true_output):
        """
        Calculates the delta for the output layer during backpropagation.

        Args:
            true_output: True target values.
        
        Returns:
            Delta value for the previous layer.
        """
        error = true_output - self.output
        self.delta = np.multiply(self.activation_derivative(self.input), error)
        self.delta = np.atleast_2d(self.delta)
        return self.calculate_delta_with_weights()
    
    def calculate_delta_with_weights(self):
        """
        Calculates the delta for the previous layer using the current layer's weights.

        Returns:
            Delta value for the previous layer.
        """
        return self.delta @ self.weights.T[:, :-1]

    def calculate_delta(self, last_layer_delta):
        """
        Calculates the delta for a hidden layer during backpropagation.

        Args:
            last_layer_delta: Delta from the subsequent layer.
        
        Returns:
            Delta value for the previous layer.
        """
        self.delta = np.multiply(self.activation_derivative(self.input), last_layer_delta)
        return self.calculate_delta_with_weights()
    
    def update_weights(self, lr):
        """
        Updates the weights of the layer using the calculated delta values.

        Args:
            lr: Learning rate for weight updates.
        """
        error = lr * np.outer(self.input_links, self.delta)
        self.weights += error

    def _apply_dropout_mask(self):
        """
        Applies dropout by randomly setting some input neurons to zero based on dropout probability.
        """
        if self.dropout_prob is not None:
            random_matrix = np.random.rand(*self.input.shape)
            dropout_mask = np.where(random_matrix < self.dropout_prob, 0, 1)
            self.input *= dropout_mask

    def _apply_batch_norm(self):
        """
        Applies batch normalization to the input data.
        """
        if self.batch_norm_1d_size is not None:
            batch = self._generate_mini_batch()
            mean, std = np.mean(batch, axis=0), np.std(batch, axis=0)
            self.input = (self.input - mean) / std

    def _generate_mini_batch(self):
        """
        Generates a mini-batch for batch normalization.

        Returns:
            A mini-batch of input data.
        """
        start_idx = np.random.randint(0, len(self.input) - self.batch_norm_1d_size + 1)
        batch = self.input[start_idx:start_idx + self.batch_norm_1d_size]
        return batch
