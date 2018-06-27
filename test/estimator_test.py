import sys
import numpy as np
import pickle
from datetime import datetime

sys.path.append('../src/')

from neural_net import NeuralNetwork

nn = NeuralNetwork(N_inputs=3,N_outputs=4, replay_memory_max_size=55, training_start=40, log_path='../Logs/' + datetime.now().strftime('%Y%m%d_%H%M%S'))

state = np.ones([5,3])
# state = np.array([[1., 1., 1.]])
train_dist = np.ones([5,4])*0.5
train_val = np.array([5., 5., 5., 5., 5.])
dist, val = nn.forward_pass(state)
print(dist)
print(val)


#From saved state and actions
with open('../Figs/estimator_input.pkl', 'rb') as f:
    state_loaded, allowed_actions_loaded = pickle.load(f)

dist_, val_ = nn.forward_pass(state_loaded)
print(dist_)
print(val_)

# for i in range(1,100000):
#     nn.forward_pass(state_loaded)


#Training
for i in range(0,5):
    nn.add_samples_to_memory(state, train_dist, train_val) #Should be state, mcts_dist, actual_value
    for i in range (0,5):
        nn.update_network()

#Save/load
dist1, val1 = nn.forward_pass(state)
replay_memory1 = nn.replay_memory
replay_memory_write_idx1 = nn.replay_memory_write_idx
replay_memory_size1 = nn.replay_memory_size

nn.save_network("../Logs/testSave2")

nn2 = NeuralNetwork(N_inputs=nn.N_inputs,N_outputs=nn.N_outputs, replay_memory_max_size=nn.replay_memory_max_size, training_start=nn.training_start, log_path='../Logs/' + datetime.now().strftime('%Y%m%d_%H%M%S'))
nn2.load_network("../Logs/testSave2")

dist2, val2 = nn2.forward_pass(state)
replay_memory2 = nn2.replay_memory
replay_memory_write_idx2 = nn2.replay_memory_write_idx
replay_memory_size2 = nn2.replay_memory_size

print(dist1 == dist2)
print(val1 == val2)
print(replay_memory1[0][0][0]==replay_memory2[0][0][0]) #Lazy comparison of just first element
print(replay_memory_write_idx1 == replay_memory_write_idx2)
print(replay_memory_size1 == replay_memory_size2)