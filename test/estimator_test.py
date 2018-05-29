import sys
import numpy as np
import pickle

sys.path.append('../src/')

from nn_estimator import NNEstimator

nn = NNEstimator(N_states=3,N_actions=4)

state = np.ones([3,3])
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


#Training
nn.update_network(state, dist_act, est_val) #Should be state, mcts_dist, actual_value

# TensorBoard(log_dir='../Logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#Save/load
nn.save_network("../Logs/testSave2")
nn.save_network("../Logs/testDir/testSave2")

nn.load_network("../Logs/testSave2")