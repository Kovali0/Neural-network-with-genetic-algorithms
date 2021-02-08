"""
File with all code for neural network
"""
import math
import numpy as np


def ReLu(x):
    return np.maximum(0, x)


def ReLu_Derivative(x):
    return np.where(x <= 0, 0, 1)


def Heaviside(x):
    return np.where(x >= 0.0,  1.0, 0.0)


def Heaviside_Derivative(x):
    return np.ones(x.shape)


def Sigmoid(x):
    return np.where(x > 0, 1 / 1 + np.exp(-x), np.exp(x) / (1 + np.exp(x)))


def Sigmoid_Derivative(x):
    return Sigmoid(x) * (1 - Sigmoid(x))


class Layer:
    """Class for one layer in neural networks, which have all weights, biases and activation functions."""
    def __init__(self, size, activation_function, input_size):
        """Init with random numbers layers with given size and activation function."
        :param size: number of neurons
        :param activation_function: activation function
        :param input_size: input data size and shape"""
        self.size = size
        self.activation_fun = activation_function
        self.input_size = input_size 
        self.weights_number = size * input_size
        self.weights = np.random.randn(size, input_size)
        self.biases = np.random.randn(size, 1)


class NeuralNetwork:
    """Genetic Neural Network Class"""
    def __init__(self):
        """Create new neural network"""
        self.l1 = Layer(2, ReLu, 2)
        self.l2 = Layer(3, ReLu, self.l1.size)
        self.l3 = Layer(2, Heaviside, self.l2.size)
        self.layers = []
        self.layers.append(self.l1)
        self.layers.append(self.l2)
        self.layers.append(self.l3)
        self.layers_number = 3
        self.accuracy = 0

    def train(self, samples, labels, epochs, batch_size=1, learning_rate=0.01, decay=0.1):
        """Train process for network
        :param samples: data
        :param labels: correct classifications for data
        :param epochs: epochs number
        :param batch_size: size of split data to packages for learning
        :param learning_rate: how much change the model in response to the estimated error
        :param decay: decay importance"""
        samples, labels = self._shuffle(samples.T, labels.T)
        n = batch_size
        batches = [(samples[:, n * i: n * (i + 1)], labels[n * i: n * (i + 1)]) for i in range(math.ceil(samples.shape[1] / n))]
        for e in range(epochs):
            self.calculate_accuracy(samples, labels)
            print("epoch_{}: {}%".format(e, self.accuracy))
            current_learning_rate = learning_rate / (1 + e * decay)
            for X, y in batches:
                next_input = X
                weighted_inputs = []
                activations = [np.copy(next_input)]
                for layer in self.layers:
                    next_input, weighted_sum = self._step_forward(next_input, layer, with_weighted_sum=True)
                    weighted_inputs.append(weighted_sum)
                    activations.append(next_input)
                dw, db = self.backpropagation(y, weighted_inputs, activations)
                for idx, layer in enumerate(self.layers):
                    layer.weights += current_learning_rate * dw[idx]
                    layer.biases += current_learning_rate * db[idx]

    def backpropagation(self, labels, weighted_inputs, activations):
        """Main backpropagation method for learning
        :param labels: correct classification
        :param weighted_inputs: input data
        :param activations: last activations
        :return: tuple of deltas, which are used to correct weights and biases"""
        deltas = [None] * len(self.layers)
        ones = np.ones(labels.shape)
        zeros = np.zeros(labels.shape)
        y = np.where(labels > 0, [ones, zeros], [zeros, ones])
        deltas[-1] = (y - activations[-1]) * Heaviside_Derivative(weighted_inputs[-1])
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.layers[i + 1].weights.T.dot(deltas[i + 1]) * ReLu_Derivative(weighted_inputs[i])
        batch_size = labels.shape[0]
        delta_bias = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
        delta_weights = [d.dot(activations[i].T) / float(batch_size) for i, d in enumerate(deltas)]
        return delta_weights, delta_bias

    def _shuffle(self, samples, labels):
        """Shuffle samples and labels in the same way
        :param samples: data
        :param labels: properly data labels
        :return: numpy array of samples and labels after shuffle"""
        indices = np.random.permutation(labels.shape[0])
        return np.array(samples[:, indices]), np.array(labels[indices])

    def _step_forward(self, samples, layer, *, with_weighted_sum=None):
        """Run neurons calculations on one layer
        :param samples: Input for neurons
        :param layer: layer which will be calculate
        :param with_weighted_sum: param, which decided if weighted sum will be returned additionally
        :return: tuple of activation value plus weighted sum"""
        weighted_sum = layer.weights.dot(samples) + layer.biases
        if with_weighted_sum:
            return layer.activation_fun(weighted_sum), weighted_sum
        else:
            return layer.activation_fun(weighted_sum)

    def predict(self, samples):
        """Simple predict a classification for sample
        :param samples: samples to predict
        :return: tuple of activation value plus weighted sum"""
        prediction = samples
        for layer in self.layers:
            prediction = self._step_forward(prediction, layer)
        return tuple(prediction)

    def calculate_accuracy(self, samples, labels):
        """Calculate neural network accuracy
        :param samples: samples to predict
        :param labels: correct labels for samples
        :return: network accuracy"""
        pred_1, pred_2 = self.predict(samples)
        pred = np.where(pred_1 > pred_2, 1, 0)
        results = np.where(pred == labels, 1, 0)
        self.accuracy = np.mean(results)
        return self.accuracy
