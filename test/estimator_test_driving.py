import sys
import numpy as np
import pickle
from datetime import datetime

sys.path.append('../src/')

from nn_estimator import NNEstimator

nn = NNEstimator(N_states=82,N_actions=5, replay_memory_max_size=55, training_start=40, log_path='../Logs/tmp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

# state = np.zeros([1,nn.N_states])
# state = np.array([list(range(1,63))])
# state[0,2] = 1
# state[0,14] = 1
# state[0,17] = 1
# nn.net.model.predict(state)
# tmp = nn.net.tmpModel.predict(state)

np.random.seed(1)
state = np.random.rand(20,nn.N_states)
# state = np.ones([20,nn.N_states])
allowed_actions = [[True, True, False, True, True],[True, False, False, True, True],[True, True, True, True, True],[False, False, False, False, True],[True, True, True, True, True],[True, True,    False, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, True,    False, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, False, True, False, True],[True, True, True, True, True],[True, True,    False, True, True],[True, False, False, True, True],[True, True, True, True, True],[True, False, False, True, True],[True, True, True, True, True]]
np.asarray(allowed_actions)*1
# state = np.zeros([1,20])
# state[0][0] = 769.604
# state[0][1] = 38.25
# state[0][2] = 19.758
# allowed_actions = [[0.0, 0.0, 0.0, 1.0, 1.0]]
train_dist = np.ones([20,nn.N_actions])*0.5
train_val = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., ])
est_val = nn.estimate_value(state)
dist_act = nn.estimate_distribution(state,allowed_actions)
print(est_val)
print(dist_act)
nn.debug_print_n_calls()

#Test batch norm
dist_act_single = nn.estimate_distribution(np.expand_dims(state[0],0),np.expand_dims(allowed_actions[0],0))
assert(np.allclose(dist_act_single[0],dist_act[0]))   #There is a small difference when calling witha  batch and when calling with a single vecotr. Does it matter?



#Training
for i in range(0,5):
    nn.add_samples_to_memory(state, train_dist, train_val) #Should be state, mcts_dist, actual_value
    for i in range (0,5):
        nn.update_network()


#Save/load
est_val1 = nn.estimate_value(state)
dist_act1 = nn.estimate_distribution(state,allowed_actions)

nn.save_network("../Logs/testSave2")
nn.load_network("../Logs/testSave2")

est_val2 = nn.estimate_value(state)
dist_act2 = nn.estimate_distribution(state,allowed_actions)

assert(np.all(est_val1 == est_val2))
assert(np.all(dist_act1 == dist_act2))


#Test batch norm
dist_act_single = nn.estimate_distribution(np.expand_dims(state[0],0),np.expand_dims(allowed_actions[0],0))
assert(np.allclose(dist_act_single[0],dist_act1[0]))