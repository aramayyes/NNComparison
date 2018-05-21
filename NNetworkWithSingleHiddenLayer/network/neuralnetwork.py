import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, numberof_input, numberof_hidden, numberof_output, learning_rate, numberof_trainings):
        self.numberof_input = numberof_input
        self.numberof_hidden = numberof_hidden
        self.numberof_output = numberof_output
        self.numberof_trainings = numberof_trainings

        self.rate = learning_rate

        self.__activation_function = lambda x: scipy.special.expit(x)

        self.weights_ih, self.weights_ho = self.__generate_random_weights()

        self.trained = False
        pass

    def __generate_random_weights(self):
        weights_ih = np.random.normal(0.0, pow(self.numberof_hidden, -0.5),
                                      (self.numberof_hidden, self.numberof_input))
        weights_ho = np.random.normal(0.0, pow(self.numberof_output, -0.5),
                                      (self.numberof_output, self.numberof_hidden))

        return weights_ih, weights_ho

    def train(self, inputs_vector, target_vector):
        inputs = np.array(inputs_vector, ndmin=2).T
        target = np.array(target_vector, ndmin=2).T

        inputof_hidden = np.dot(self.weights_ih, inputs)
        outputof_hidden = self.__activation_function(inputof_hidden)

        inputof_final = np.dot(self.weights_ho, outputof_hidden)
        outputof_final = self.__activation_function(inputof_final)

        # calculate the errors for each layer
        errorof_output = target - outputof_final
        errorof_hidden = np.dot(self.weights_ho.T, errorof_output)

        # update the weights
        self.weights_ho += self.rate * np.dot(errorof_output * outputof_final * (1.0 - outputof_final),
                                              outputof_hidden.T)
        self.weights_ih += self.rate * np.dot(errorof_hidden * outputof_hidden * (1.0 - outputof_hidden), inputs.T)
        pass

    def predict(self, inputs_vector):
        inputs = np.array(inputs_vector, ndmin=2).T

        inputof_hidden = np.dot(self.weights_ih, inputs)
        outputof_hidden = self.__activation_function(inputof_hidden)

        inputof_final = np.dot(self.weights_ho, outputof_hidden)
        outputof_final = self.__activation_function(inputof_final)
        return outputof_final

    def log(self, only_weights=True):
        if not only_weights:
            print('Number of input nodes:', self.numberof_input)
            print('Number of hidden nodes:', self.numberof_hidden)
            print('Number of output nodes:', self.numberof_output)

            print('Learning rate:', self.rate)

            print('Number of trainings:', self.numberof_trainings)

        print('Weights: Input -> Hidden\n', self.weights_ih)
        print('Weights: Hidden -> Output\n', self.weights_ho)
        pass

    def save_weights(self, name):
        np.save(name + '_ih', self.weights_ih)
        np.save(name + '_ho', self.weights_ho)

    def load_weights(self, name):
        self.weights_ih = np.load(name + '_ih.npy')
        self.weights_ho = np.load(name + '_ho.npy')
