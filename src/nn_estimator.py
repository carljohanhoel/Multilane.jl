import sys
import os
import numpy as np
import pickle

from neural_net import AGZeroModel

class NNEstimator:
    def __init__(self):
        self.N_states = 2
        self.N_actions = 4
        self.net = AGZeroModel(self.N_states, self.N_actions)
        self.net.create_simple()
        self.n_val_calls = 0
        self.n_prob_calls = 0

    def estimate_value(self, state):
        # value = 0.0
        self.n_val_calls+=1
        [probabilities, value] = self.net.predict(state)
        return np.float64(value)   #Conversion required for julia code. NN outputs array with one element.

    def estimate_distribution(self, state, allowed_actions):
        # n_actions = len(possible_actions)
        # probabilities = np.ones(n_actions)*1/n_actions
        self.n_prob_calls+=1
        [dist, value] = self.net.predict(state)

        dist = dist*allowed_actions
        sum_dist = np.sum(dist,axis=1)
        dist = [dist[i,:]/sum_dist[i] for i in range(0,len(sum_dist))]
        return np.float64(dist)   #Float64 required in julia code. NN outputs float32.

    def debug_save_input(self, state, possible_actions):
        print("in debug_save_input")
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/Figs/"   #Necessary with absolute path when calling from julia tests

        print(state)
        print(possible_actions)

        with open(dir_path + 'estimator_input.pkl', 'wb') as f:
            pickle.dump([state, possible_actions], f)

    def debug_print_n_calls(self):
        print("Number of value calls: ", self.n_val_calls)
        print("Number of probability calls: ", self.n_prob_calls)



