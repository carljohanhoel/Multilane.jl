import sys
import numpy as np
import pickle
from datetime import datetime
from time import sleep

sys.path.append('../src/')

from nn_estimator import NNEstimator

nn = NNEstimator(N_states=3,N_actions=4, replay_memory_max_size=55, training_start=40, log_path='../Logs/' + datetime.now().strftime('%Y%m%d_%H%M%S'))

# state_batch = np.ones([20,3])
# allowed_actions_batch = [[True, True, False, True],[True, False, False, True],[True, True, True, True],[True, False, False, True],[True, True, True, True],[True, True,    False, True],[True, False, False, True],[True, True, True, True],[True, False, False, True],[True, True, True, True],[True, True,    False, True],[True, False, False, True],[True, True, True, True],[True, False, False, True],[True, True, True, True],[True, True,    False, True],[True, False, False, True],[True, True, True, True],[True, False, False, True],[True, True, True, True]]
state = np.array([[1., 1., 1.]])
allowed_actions = [[True,True,True,True]]
train_state = np.ones([5,3])
train_dist = np.ones([5,4])*0.5
train_val = np.array([5., 5., 5., 5., 5.])
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

#Save/load
nn.save_network("../Logs/testDir/testSave3")

#Training
for i in range (0,1000):
    if i==55:
        debug = 1
    nn.update_network(train_state, train_dist, train_val) #Should be state, mcts_dist, actual_value

est_val = nn.estimate_value(state)
dist_act = nn.estimate_distribution(state,allowed_actions)

sleep(5)
nn.terminate()

load_network = "../Logs/testDir/testSave2"
nn_loaded = NNEstimator(N_states=3,N_actions=4, replay_memory_max_size=55, training_start=40, log_path='../Logs/' + datetime.now().strftime('%Y%m%d_%H%M%S'), load_network=load_network)

est_val = nn_loaded.estimate_value(state)
dist_act = nn_loaded.estimate_distribution(state,allowed_actions)
print(est_val)
print(dist_act)

sleep(5)
nn_loaded.terminate()