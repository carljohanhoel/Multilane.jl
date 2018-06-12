import sys
import numpy as np
import pickle
from datetime import datetime

sys.path.append('../src/')

from nn_estimator import NNEstimator

nn = NNEstimator(N_states=20,N_actions=5, replay_memory_max_size=55, training_start=40, log_path='../Logs/' + datetime.now().strftime('%Y%m%d_%H%M%S'))

state = np.ones([20,20])
allowed_actions = [[True, True, False, True, True],[True, False, False, True, True],[True, True, True, True, True],[False, False, False, False, True],[True, True, True, True, True],[True, True,    False, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, True,    False, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, False, True, False, True],[True, True, True, True, True],[True, True,    False, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, False, False, True, True],[True, True, True, True, True]]
np.asarray(allowed_actions)*1
# state = np.zeros([1,20])
# state[0][0] = 769.604
# state[0][1] = 38.25
# state[0][2] = 19.758
# allowed_actions = [[0.0, 0.0, 0.0, 1.0, 1.0]]
train_dist = np.ones([20,5])*0.5
train_val = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., ])
est_val = nn.estimate_value(state)
dist_act = nn.estimate_distribution(state,allowed_actions)
print(est_val)
print(dist_act)
nn.debug_print_n_calls()



#Training
for i in range(0,5):
    nn.add_samples_to_memory(state, train_dist, train_val) #Should be state, mcts_dist, actual_value
    for i in range (0,5):
        nn.update_network()

#Save/load
nn.save_network("../Logs/testSave2")
nn.save_network("../Logs/testDir/testSave2")

nn.load_network("../Logs/testSave2")