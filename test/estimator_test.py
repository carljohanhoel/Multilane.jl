import sys
import numpy as np
import pickle

sys.path.append('../src/')

from nn_estimator import NNEstimator

nn = NNEstimator(N_states=2,N_actions=4)

state = np.ones([3,2])
allowed_actions = [[True, True, False, True],[True, False, False, True],[True, True, True, True]]
est_val = nn.estimate_value(state)
dist_act = nn.estimate_distribution(state,allowed_actions)
print(est_val)
print(dist_act)
nn.debug_print_n_calls()



#From saved state and actions
with open('../Figs/estimator_input.pkl', 'rb') as f:
    state_loaded, allowed_actions_loaded = pickle.load(f)

est_val_loaded = nn.estimate_value(state_loaded)
dist_act_loaded = nn.estimate_distribution(state_loaded,allowed_actions_loaded)
print(est_val_loaded)
print(dist_act_loaded)

# for i in range(1,100000):
#     nn.estimate_value(state_loaded)

