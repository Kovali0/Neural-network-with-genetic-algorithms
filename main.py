"""
Main program
"""
import random
import numpy as np
import copy
from neural_network import NeuralNetwork
# from keras.models import Sequential
# from keras.layers import Dense


def generate_samples(size, mean_x, mean_y, standard_deviation_x, standard_deviation_y):
    return np.random.default_rng().normal((mean_x, mean_y), (standard_deviation_x, standard_deviation_y), (size, 2))


# Globals
samples = {0: [], 1: []}
mean_x = [-1, 4, 1, 7]
mean_y = 2
std_dev = 1
for x in range(4):
    if x < len(mean_x) // 2:
        label = 0
    else:
        label = 1
    samples[label].extend(generate_samples(100, mean_x[x], mean_y, std_dev, std_dev))

for k, v in samples.items():
    samples[k] = np.array(v)

x_train = np.concatenate([s for l, s in samples.items()])
y_train = np.concatenate([np.repeat(l, s.shape[0]) for l, s in samples.items()])
population = 50
mutation_chance = 2
top_pick = 10


# def init_keras_model():
#     model = Sequential()
#     model.add(Dense(2, input_dim=8, activation='relu'))
#     model.add(Dense(3, activation='relu'))
#     model.add(Dense(2, activation='sigmoid'))
#     return model


def main():
    """Main with genetic algorithm loops for neural networks.
    Print results. """
    generation = 0
    best_accuracy = 0.0
    networks = [NeuralNetwork() for _ in range(population)]
    best_weights = []
    best_biases = []

    while best_accuracy < 0.9 and generation < 100:
        generation += 1
        print("========== Generation number ", generation, " ==========")

        for nn in networks:
            current_accuracy = nn.calculate_accuracy(x_train.T, y_train)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Best Accuracy: ', best_accuracy)
                best_weights.clear()
                best_biases.clear()
                for layer in nn.layers:
                    best_weights.append(layer.weights)
                    best_biases.append(layer.biases)

        networks = sorted(networks, key=lambda z: z.accuracy, reverse=True)
        print(networks[0].layers[0].weights)

        new_generation = []
        for i in range(top_pick):
            for j in range(population//top_pick):
                nn1 = copy.deepcopy(networks[i])
                nn2 = copy.deepcopy(networks[random.randint(0, top_pick)])
                locus = random.randint(0, 1)
                for idx, layer in enumerate(nn1.layers):
                    for index, neuron in enumerate(layer.weights):
                        tmp = neuron[locus]
                        neuron[locus] = nn2.layers[idx].weights[index][locus]
                        nn2.layers[idx].weights[index][locus] = tmp
                        if random.randint(0, 100) < mutation_chance:
                            # print("MUTATION!")
                            # layer.weights[locus] = np.negative(layer.weights[locus])
                            # layer.weights[locus] = np.random.randn(np.size(layer.weights[locus]))
                            neuron[locus] = np.random.randn()
                new_generation.append(nn1)
                new_generation.append(nn2)
        networks.clear()
        networks = copy.deepcopy(new_generation)

    print("Selection accuracy: ")
    print(best_accuracy)

    genetic_nn = NeuralNetwork()
    for idx, layer in enumerate(genetic_nn.layers):
        layer.weights = best_weights[idx]
        layer.biases = best_biases[idx]
    genetic_nn.train(x_train, y_train, 10, 10)
    genetic_nn.calculate_accuracy(x_train.T, y_train)

    print("Prediction accuracy: ")
    print(genetic_nn.accuracy)


if __name__ == '__main__':
    main()
