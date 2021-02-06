"""
File with all code for neural network
"""
import numpy as np


def ReLu(x):
    return np.maximum(0, x)


def ReLu_Derivative(x):
    return 0 if x <= 0 else 1


class Layer:
    def __init__(self, size, activation_function, input_size):
        self.size = size
        self.activation_fun = activation_function
        self.input_size = input_size
        self.weights_number = size * input_size
        self.weights = [np.random.randn(input_size) for _ in range(size)]
        self.biases = [np.random.randn() for _ in range(size)]


class NeuralNetwork:
    def __init__(self):
        self.l1 = Layer(2, 0, 2)
        self.l2 = Layer(3, 0, self.l1.size)
        self.l3 = Layer(2, 0, self.l2.size)
        self.layers = []
        self.layers.append(self.l1)
        self.layers.append(self.l2)
        self.layers.append(self.l3)
        self.layers_number = 3
        self.accuracy = 0

    def train(self, samples, labels, epochs):
        pass

    def forward_propagation(self, samples, layer):
        activation_output = []
        for i, neuron_weights in enumerate(layer.weights):
            a = neuron_weights.dot(samples) + layer.biases[i]
            activation_output.append(ReLu(a))
        return activation_output

    def predict(self, samples):
        prediction = samples
        for layer in self.layers:
            prediction = self.forward_propagation(prediction, layer)
        return tuple(prediction)
