import numpy as np
import random
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt


class MnistdataManager:
    def __init__(self):
        self.training_data_file_path = 'mnist/files/mnist_train.csv'
        self.training_data_list = None

        self.testing_data_file_path = 'mnist/files/mnist_test.csv'
        self.testing_data_list = None
        pass

    # load the MNIST data for training
    def __load_training_data(self):
        print('Loading training data...')

        training_data_file = open(self.training_data_file_path, 'r')
        self.training_data_list = training_data_file.readlines()
        training_data_file.close()

        print('Loading finished.', len(self.training_data_list), 'records are loaded.')
        pass

    # load the MNIST dat for testing
    def __load_testing_data(self):
        print('Loading testing data...')

        testing_data_file = open(self.testing_data_file_path, 'r')
        self.testing_data_list = testing_data_file.readlines()
        testing_data_file.close()

        print('Loading finished.', len(self.testing_data_list), 'records are loaded.')
        pass

    # train the given neural network
    def train(self, network):
        if self.training_data_list is None:
            self.__load_training_data()

        print('Training the neural network...')

        start_time = timer()

        for r in range(network.numberof_trainings):
            print('Epoch', r + 1)
            for record in self.training_data_list:
                values = record.split(',')

                label = int(values[0])

                input_values = np.asfarray(values[1:])
                # map the values from the range [0, 255] to the range [0.01, 1]
                inputs = (input_values / 255 * 0.99) + 0.01

                # create the target output vector
                targets = np.zeros(10) + 0.01
                targets[label] = 0.99

                network.train(inputs, targets)

        network.trained = True
        end_time = timer()

        print('Training finished in', end_time - start_time)
        pass

    # test the given neural network
    def test(self, network, plot_needed=True):
        if self.testing_data_list is None:
            self.__load_testing_data()

        print('Testing...')

        # Select a random record from testing_database
        random_record = random.choice(self.testing_data_list)
        values = random_record.split(',')

        correct_label = int(values[0])
        print('The correct answer is', correct_label)

        input_values = np.asfarray(values[1:])
        # map the values from the range [0, 255] to the range [0.01, 1]
        inputs = (input_values / 255 * 0.99) + 0.01

        # Get the network's answer
        output = network.predict(inputs)
        network_answer = np.argmax(output)

        if plot_needed:
            fig = plt.figure(figsize=(12, 4))

            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(inputs.transpose().reshape(28, 28),
                      cmap=matplotlib.cm.binary, interpolation='None')

            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            ax = fig.add_subplot(1, 2, 2)
            bar_list = ax.bar(np.arange(10), output.T[0], align='center')
            bar_list[network_answer].set_color('g')
            ax.set_xticks(np.arange(10))
            ax.set_xlim([-1, 10])
            ax.grid('on')

            plt.show()
            pass

        print('The network\'s answer is', network_answer)
        print('Result', '+' if network_answer == correct_label else '-')
        print('Testing finished')

        pass

    # calculate accuracy of the given network in percentage
    def calculate_accuracy(self, network):
        if self.testing_data_list is None:
            self.__load_testing_data()

        print('Getting the accuracy...')

        numberof_correct_answers = 0
        numberof_all_answers = 0

        for record in self.training_data_list:
            values = record.split(',')

            correct_label = int(values[0])

            input_values = np.asfarray(values[1:])
            # map the values from the range [0, 255] to the range [0.01, 1]
            inputs = (input_values / 255 * 0.99) + 0.01

            # Get the network's answer
            output = network.predict(inputs)
            network_answer = np.argmax(output)

            numberof_all_answers += 1
            if correct_label == network_answer:
                numberof_correct_answers += 1

        # calculate the accuracy
        print('Number of all answers:', numberof_all_answers)
        print('Number of correct answers:', numberof_correct_answers)
        print('Accuracy:', numberof_correct_answers / numberof_all_answers * 100, '%')

        return numberof_correct_answers / numberof_all_answers * 100
        pass
