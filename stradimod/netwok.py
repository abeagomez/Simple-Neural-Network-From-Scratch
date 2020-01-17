import numpy as np
from .utils import *
from .layers import Dense


class Network():

    def __init__(self):
        self.layers = []
        self.model_length = 0

    def add(self, layer):
        """
        Adds new layer at the end of Network

        Arguments:
            layer { Dense } -- [New Layer]
        """
        self.model_length += 1
        layer.set_name(self.model_length)
        self.layers.append(layer)

    def build(self, input_size):
        """
        Initializes layers in Network
        Arguments:
            input_size {int} -- [input data dimension]
        """
        self.__init_layers(input_size)

    def train(self, x_train, y_train, learning_rate=0.01, epochs=10):
        """
        Trains Network

        Arguments:
            x_train {np_matrix} -- [Training input set]
            y_train {np_matrix} -- [Training output set]

        Keyword Arguments:
            learning_rate {float} -- [Netwrok learning rate] (default: {0.01})
            epochs {int} -- [No. of epochs for training] (default: {10})
        """
        for epoch in range(epochs):
            self.__forward_activation(x_train)
            AL = self.layers[-1].A
            if epoch % 100 == 0:
                loss = self.__get_loss(AL, y_train)
                print(f"==== epoch no. {epoch} =====  loss {loss} ====")
            self.__backpropagation(AL, y_train, x_train)
            self.__update_layers_parameters(learning_rate)

    def predict(self, x_input, y_target, threshold=0.5):
        """[summary]

        Arguments:
            x_input { np_matrix } -- [Testing input set]
            y_target { np_matrix } -- [Testing output set]

        Keyword Arguments:
            threshold {float} -- [threshold value for binary segmentation] (default: {0.5})
        """
        m = x_input.shape[1]
        self.__forward_activation(x_input)
        output = self.layers[-1].A
        predicted = np.where(output > threshold, 1, 0)
        accuracy = np.sum(predicted == y_target) / m
        return(accuracy)

    def __init_layers(self, input_size):
        for layer in self.layers:
            layer.build(input_size)
            input_size = layer.size

    def __update_layers_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def __forward_activation(self, layer_input):
        for layer in self.layers:
            layer_input = layer.forward_activation(layer_input)
            layer_input = layer.A

    def __backpropagation(self, AL, Y, X):
        number_of_layers = len(self.layers)
        dAL = crossentropy(Y, AL, 0)
        dA_right = dAL

        for l_index in reversed(range(0, number_of_layers)):
            activation = self.layers[l_index].activation
            W_curr = self.layers[l_index].W
            Z_curr = self.layers[l_index].Z
            if l_index - 1 >= 0:
                A_left = self.layers[l_index - 1].A
            else:
                A_left = X
            dA_right, dW_curr, db_curr = self.__single_layer_backward(
                dA_right, Z_curr, W_curr, A_left, activation)

            self.layers[l_index].update_gradients(dW_curr, db_curr)

    def __single_layer_backward(self, dA_right, Z, W, A_left, activation, log=False):
        m = A_left.shape[1]
        dZ = activation(Z, dA_right, 0)

        dW = np.dot(dZ, A_left.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_left = np.dot(W.T, dZ)

        if log:
            print("Single Layer Backward")
            print(f"Z shape is {Z.shape}")
            print(f"dA_right shape is {dA_right.shape}")
            print(f"W shape is {W.shape}")
            print(f"dZ shape is {dZ.shape}")
            print(f"dA_left shape is {dA_left.shape}")

        return dA_left, dW, db

    def __get_loss(self, output, target):
        return crossentropy(target, output)
