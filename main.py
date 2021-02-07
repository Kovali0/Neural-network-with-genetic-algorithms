"""
Main program
"""
import random
import numpy as np
import copy
from neural_network import NeuralNetwork, Layer


def generate_samples(size, mean_x, mean_y, standard_deviation_x, standard_deviation_y):
    return np.random.default_rng().normal((mean_x, mean_y), (standard_deviation_x, standard_deviation_y), (size, 2))


# Globals
samples = {0: [], 1: []}
mean_x = [3, 7]
mean_y = 2
std_dev = 1
for x in range(2):
    if x < len(mean_x) // 2:
        label = 0
    else:
        label = 1
    samples[label].extend(generate_samples(50, mean_x[x], mean_y, std_dev, std_dev))

for k, v in samples.items():
    samples[k] = np.array(v)

# Globals
x_train = np.concatenate([s for l, s in samples.items()])
y_train = np.concatenate([np.repeat(l, s.shape[0]) for l, s in samples.items()])
population = 55
mutation_chance = 1


def main():
    generation = 0
    best_accuracy = 0.0
    networks = [NeuralNetwork() for _ in range(population)]
    optimal_weights = []

    while best_accuracy < 0.9 and generation < 100:
        generation += 1
        print("========== Generation number ", generation, " ==========")

        for nn in networks:
            good_prediction = 0
            for i, x in enumerate(x_train):
                pred_1, pred_2 = nn.predict(x)
                if pred_1 > pred_2 and y_train[i] == 1 or pred_1 < pred_2 and y_train[i] == 0:
                    good_prediction += 1
            nn.accuracy = good_prediction / np.size(x_train)
            if nn.accuracy > best_accuracy:
                best_accuracy = nn.accuracy
                print('Best Accuracy: ', best_accuracy)

        networks = sorted(networks, key=lambda z: z.accuracy, reverse=True)
        print(networks[0].layers[0].weights)
        for layer in networks[0].layers:
            optimal_weights.append(layer.weights)

        new_generation = []
        for i in range(5):
            new_generation.append(copy.deepcopy(networks[i]))
            for j in range(5):
                nn1 = copy.deepcopy(networks[i])
                nn2 = copy.deepcopy(random.choice(networks))
                locus = random.randint(0, 1)
                for idx, layer in enumerate(nn1.layers):
                    tmp = layer.weights[locus]
                    layer.weights[locus] = nn2.layers[idx].weights[locus]
                    nn2.layers[idx].weights[locus] = tmp
                    if random.randint(0, 100) == mutation_chance:
                        print("MUTATION!")
                        # layer.weights[locus] = np.negative(layer.weights[locus])
                        layer.weights[locus] = np.random.randn(np.size(layer.weights[locus]))
                new_generation.append(nn1)
                new_generation.append(nn2)
        networks.clear()
        networks = new_generation

    print("Over 0.90 accuracy")
    print(best_accuracy)


if __name__ == '__main__':
    main()
