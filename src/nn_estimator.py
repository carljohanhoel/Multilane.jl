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

    def estimate_value(self, state):
        # value = 0.0
        [probabilities, value] = self.net.predict(state)
        return value

    def estimate_probabilities(self, state, possible_actions):
        # n_actions = len(possible_actions)
        # probabilities = np.ones(n_actions)*1/n_actions
        [probabilities, value] = self.net.predict(state)
        return(probabilities)

    def debug_save_input(self, state, possible_actions):
        print("in debug_save_input")
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/Figs/"   #Necessary with absolute path when calling from julia tests

        print(state)
        print(possible_actions)

        with open(dir_path + 'estimator_input.pkl', 'wb') as f:
            pickle.dump([state, possible_actions], f)




