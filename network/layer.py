import numpy as np


class Layer:

    def __init__(self, in_features, out_features, activation_function, activation_derivative, dropout_prob):
        self.weights = np.random.uniform(low=-1, high=1, size=(in_features, out_features))
        self.activation_function, self.activation_derivative = activation_function, activation_derivative
        self.input_links = None
        self.input = None
        self.output = None
        self.delta = None
        self.dropout_prob = dropout_prob

    def train_forward(self, input_links):
        self.input_links = input_links
        self.input = input_links @ self.weights

        self.apply_dropout_mask() # Dropout

        self.output = self.activation_function(self.input)
        return self.output

    def forward(self, input_links):
        return self.activation_function(input_links @ self.weights)

    def calculate_output_delta(self, true_output):
        error = (true_output - self.output)
        self.delta = np.multiply(self.activation_derivative(self.input), error)
        self.delta = np.atleast_2d(self.delta)
        return self.calculate_delta_with_weights()
    
    def calculate_delta_with_weights(self):
        return self.delta @ self.weights.T

    def calculate_delta(self, last_layer_delta):
        self.delta = np.multiply(self.activation_derivative(self.input), last_layer_delta)
        return self.calculate_delta_with_weights()

    def update_weights(self, lr):
        error = lr * np.outer(self.input_links, self.delta)
        self.weights += error

    def apply_dropout_mask(self):
        if self.dropout_prob != None:
            random_matrix = np.random.rand(*self.input.shape)
            dropout_mask = np.where(random_matrix < self.dropout_prob, 0, 1)
            self.input *= dropout_mask