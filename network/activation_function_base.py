import numpy as np
from scipy.special import expit

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