import numpy as np
import utils
from layers import Dense


class Model():
    def __init__(self):
        self.layers = []
        self.model_length = 0

    def add(self, layer):
        self.model_length += 1
        layer.set_name(self.model_length)
        self.layers.append(layer)

    def build(self, input_size):
        self.__init_layers(input_size)

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

    def __get_loss(self, output, target):
        return utils.crossentropy(target, output)
