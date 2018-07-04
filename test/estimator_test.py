import sys
import numpy as np
import pickle
from datetime import datetime
import time

sys.path.append('../src/')

from neural_net import NeuralNetwork

nn = NeuralNetwork(N_inputs=3,N_outputs=4, replay_memory_max_size=1052, training_start=1040, log_path='../Logs/tmp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

rng = np.random.RandomState(1)
#
state = rng.rand(200,3)
train_dist = rng.rand(200,4)
train_dist = train_dist/np.sum(train_dist,1)[:,None]
train_val = rng.rand(200,1)
# state = rng.rand(1,3)
# train_dist = rng.rand(1,4)
# train_dist = train_dist/np.sum(train_dist)
# train_val = rng.rand(1,1)
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

# start_time = time.time()
# #Training
# for i in range(0,100):
#     nn.add_samples_to_memory(state, train_dist, train_val) #Should be state, mcts_dist, actual_value
#     for i in range (0,10):
#         nn.update_network()
# print(time.time()-start_time)

# start_time = time.time()
# for i in range(0,10000):
#     nn.model.train_on_batch(state[0:32], [train_dist[0:32], train_val[0:32]])
# print(time.time()-start_time)

start_time = time.time()
for i in range(0,5000):
    nn.model.predict(state[0:16,:])
print(time.time()-start_time)


#Save/load
dist1, val1 = nn.forward_pass(state)
rm1 = nn.rm
rm_write_idx1 = nn.rm.write_idx
rm_size1 = nn.rm.size

nn.save_network("../Logs/testSave2")

nn2 = NeuralNetwork(N_inputs=nn.N_inputs,N_outputs=nn.N_outputs, replay_memory_max_size=nn.rm.max_size, training_start=nn.rm.training_start, log_path='../Logs/' + datetime.now().strftime('%Y%m%d_%H%M%S'))
nn2.load_network("../Logs/testSave2")

dist2, val2 = nn2.forward_pass(state)
rm2 = nn2.rm
rm_write_idx2 = nn2.rm.write_idx
rm_size2 = nn2.rm.size

print(dist1 == dist2)
print(val1 == val2)
print(rm1.states[0][0]==rm2.states[0][0]) #Lazy comparison of just first element
print(rm_write_idx1 == rm_write_idx2)
print(rm_size1 == rm_size2)