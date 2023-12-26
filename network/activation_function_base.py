import numpy as np
from scipy.special import expit


# get activation function
def get_activation_function(type):
    if type == 'logistic':
        return (logistic, logistic_derivative)
    if type == 'identity':
        return (identity, identity_derivative)
    if type == 'relu':
        return (relu, relu_derivative)
    if type == 'softmax':
        return (softmax, softmax_derivative)
    else:
        raise ValueError(f"Invalid activation function type: {type}")



# hard threshold
def hard_threshold(x):
    return np.where(x > 0, 1, 0)

def hard_threshold_derivative(x):
    return np.zeros_like(x)


# logistic
def logistic(x):
    return expit(x)

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

# identity
def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)


#ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)