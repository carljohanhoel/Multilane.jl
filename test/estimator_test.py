import sys
import numpy as np
import pickle

sys.path.append('../src/')

from nn_estimator import NNEstimator

nn = NNEstimator()

state = np.array([1.0, 1.0])
state = np.ones([1,2])
possible_actions = [1,2,3,4]
est_val = nn.estimate_value(state)
prob_act = nn.estimate_probabilities(state,possible_actions)
print(est_val)
print(prob_act)


#From saved state and actions
with open('../Figs/estimator_input.pkl', 'rb') as f:
    state_loaded, possible_actions_loaded = pickle.load(f)

est_val_loaded = nn.estimate_value(state_loaded)
prob_act_loaded = nn.estimate_probabilities(state_loaded,possible_actions_loaded)
print(est_val_loaded)
print(prob_act_loaded)
