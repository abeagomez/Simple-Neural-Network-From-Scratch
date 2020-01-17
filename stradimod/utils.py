import numpy as np


def sigmoid(Z, dA=None, flag=1):
    '''
    Z: Logists' values
    dA: Gradient of layer A
    '''
    if flag:
        return sigmoid_forward(Z)
    return sigmoid_derivative(Z, dA)


def sigmoid_forward(Z):
    return 1/(1+np.exp(-Z))


def sigmoid_derivative(Z, dA):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


def relu(Z, dA=None, flag=1):
    '''
    Z: Logists' values
    dA: Gradient of layer A
    '''
    if flag:
        return relu_forward(Z)
    return relu_derivative(Z, dA)


def relu_forward(Z):
    return np.maximum(0, Z)


def relu_derivative(Z, dA):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def crossentropy(target_y, output_x, flag=1):
    '''
    target_y: Actual label vector (1, no. of examples)
    output_x: Predicted Output (1, no. of examples)
    '''
    if flag:
        return crossentropy_forward(target_y, output_x)
    return crossentropy_derivative(target_y, output_x)


def crossentropy_forward(target_y, output_x):
    '''Computes the Loss between predicted and true label
    target_y: Actual label vector (1, no. of examples)
    output_x: Predicted Output (1, no. of examples)
    '''
    m = target_y.shape[1]
    cost = np.add(np.multiply(target_y, np.log(output_x)),
                  np.multiply(1-target_y, np.log(1-output_x)))
    cost = (-1/m) * np.sum(cost, axis=1)
    return cost


def crossentropy_derivative(target_y, output_x):
    '''
    target_y: Actual label vector (1, no. of examples)
    output_x: Predicted Output (1, no. of examples)
    '''
    d = - (np.divide(target_y, output_x) -
           np.divide(1 - target_y, 1 - output_x))
    return d
