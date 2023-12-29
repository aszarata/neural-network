import numpy as np
import activation_function_base

class Layer:

    def __init__(self, 
                 in_features, 
                 out_features, 
                 activation_function='logistic', 
                 dropout_prob=None,
                 batch_norm_1d_size=None):
        

        self.weights = np.random.uniform(low=-1, high=1, size=(in_features+1, out_features))
        self.activation_function, self.activation_derivative = activation_function_base.get_activation_function(activation_function)
        self.input_links = None
        self.input = None
        self.output = None
        self.delta = None

        self.dropout_prob = dropout_prob
        self.batch_norm_1d_size = batch_norm_1d_size

        self.bias = 1

    
    # Forward
    def train_forward(self, input_links):
        self.input_links = np.append(input_links, self.bias) # bias
        self.input = self.input_links @ self.weights

        self._apply_dropout_mask() # Dropout
        self._apply_batch_norm() # Batch normalisation

        self.output = self.activation_function(self.input)
        return self.output

    def forward(self, input_links):
        bias_column = np.full((input_links.shape[0], 1), self.bias)
        input_links = np.hstack((input_links, bias_column))
        return self.activation_function(input_links @ self.weights)

    # Backward
    def calculate_output_delta(self, true_output):
        error = (true_output - self.output)
        self.delta = np.multiply(self.activation_derivative(self.input), error)
        self.delta = np.atleast_2d(self.delta)
        return self.calculate_delta_with_weights()
    
    def calculate_delta_with_weights(self):
        return self.delta @ self.weights.T[:, :-1]

    def calculate_delta(self, last_layer_delta):
        self.delta = np.multiply(self.activation_derivative(self.input), last_layer_delta)
        return self.calculate_delta_with_weights()
    

    # Update weight
    def update_weights(self, lr):
        error = lr * np.outer(self.input_links, self.delta)
        self.weights += error


    # Dropout
    def _apply_dropout_mask(self):
        if self.dropout_prob != None:
            random_matrix = np.random.rand(*self.input.shape)
            dropout_mask = np.where(random_matrix < self.dropout_prob, 0, 1)
            self.input *= dropout_mask

    # Batch normalisation
    def _apply_batch_norm(self):
        if self.batch_norm_1d_size != None:
            batch = self._generate_mini_batch()
            mean, std = np.mean(batch, axis=0), np.std(batch, axis=0)

            self.input = (self.input - mean)/std


    def _generate_mini_batch(self):
        start_idx = np.random.randint(0, len(self.input) - self.batch_norm_1d_size + 1)

        batch = self.input[start_idx:start_idx + self.batch_norm_1d_size]

        return batch