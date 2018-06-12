import sys
import os
import numpy as np
import pickle

from neural_net import AGZeroModel

class NNEstimator:
    def __init__(self, N_states, N_actions, replay_memory_max_size, training_start, log_path="./"):
        self.N_states = N_states
        self.N_actions = N_actions
        self.replay_memory_max_size = replay_memory_max_size
        self.training_start = training_start
        self.net = AGZeroModel(self.N_states, self.N_actions, self.replay_memory_max_size, self.training_start, log_path)
        # self.net.create_simple()
        self.net.create_convnet2()
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

        #DEBUG
        if len(np.asarray(allowed_actions).shape) == 1: #just one state
            n_allowed_actions = np.sum(np.asarray(allowed_actions) * 1)
            if not n_allowed_actions:
                print("error, no allowed actions")
        else:
            n_allowed_actions = np.sum(np.asarray(allowed_actions)*1,axis=1)
            if not all(n_allowed_actions):
                print("error, no allowed actions")
        if any(np.sum(dist,axis=1)==0):
            print("error, sum dist = 0")
        if np.isnan(dist).any():
            print("dist nan\n")
            print(state)

        dist = dist*allowed_actions
        sum_dist = np.sum(dist,axis=1)
        if any(sum_dist==0):   #Before the network is trained, the only allowed actions could get prob 0. In that case, set equal prior prob.
            print("error, sum allowed dist = 0")
            print(state)
            print(dist)
            print(allowed_actions)
            add_dist = ((dist*0+1) * (sum_dist == 0.)[:,np.newaxis])*allowed_actions
            dist += add_dist
            sum_dist += np.sum(add_dist,axis=1)

        # dist = [dist[i,:]/sum_dist[i] for i in range(0,len(sum_dist))]
        dist = dist/sum_dist[:,np.newaxis]
        return np.float64(dist)   #Float64 required in julia code. NN outputs float32.

    def add_samples_to_memory(self, states, dists, vals):
        self.net.add_samples_to_memory(states, dists, vals)

    def update_network(self):
        self.net.update_network()

    def save_network(self, name):
        directory = os.path.dirname(name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.net.save(name)

    def load_network(self, name):
        self.net.load(name)
        print("Net loaded: "+name)


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



