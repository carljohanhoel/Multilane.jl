import sys
import numpy as np
import pickle
from datetime import datetime

sys.path.append('../src/')

from neural_net import NeuralNetwork

nn = NeuralNetwork(N_inputs=82,N_outputs=5, replay_memory_max_size=300, training_start=200, log_path='../Logs/tmp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

n_samples = 200
np.random.seed(1)
state = np.random.rand(n_samples,nn.N_inputs)
train_dist = np.ones([n_samples,nn.N_outputs])*0.5
train_val = np.ones(n_samples)
dist_act, est_val = nn.forward_pass(state)
print(est_val)
print(dist_act)

#Test batch norm
dist_act_single, _ = nn.forward_pass(np.expand_dims(state[0],0))
assert(np.allclose(dist_act_single[0],dist_act[0]))   #There is a small difference when calling with a batch and when calling with a single vector. Does it matter?


#Training
for i in range(0,5):
    nn.add_samples_to_memory(state, train_dist, train_val) #Should be state, mcts_dist, actual_value
    for i in range (0,5):
        nn.update_network()


#Save/load
dist_act1, est_val1 = nn.forward_pass(state)

nn.save_network("../Logs/testSave2")
nn2 = NeuralNetwork(N_inputs=nn.N_inputs,N_outputs=nn.N_outputs, replay_memory_max_size=nn.replay_memory_max_size, training_start=nn.training_start, log_path='../Logs/tmp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
nn2.load_network("../Logs/testSave2")

dist_act2, est_val2 = nn2.forward_pass(state)

assert(np.all(est_val1 == est_val2))
assert(np.all(dist_act1 == dist_act2))


#Test batch norm
dist_act_single, _ = nn.forward_pass(np.expand_dims(state[0],0))
assert(np.allclose(dist_act_single[0],dist_act1[0]))