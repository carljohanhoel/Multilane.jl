import sys
import numpy as np
import pickle
from datetime import datetime
import time
import copy

sys.path.append('../src/')

from neural_net import NeuralNetwork

# N_ego_states = 3
N_ego_states = 5
N_other_vehicle_states = 4
N_other_vehicles = 20
N_inputs = N_ego_states + N_other_vehicle_states*N_other_vehicles

nn = NeuralNetwork(N_inputs=N_inputs,N_outputs=5, replay_memory_max_size=300, training_start=200, log_path='../Logs/tmp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

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

#Test maxpool, that vehicles are interchangeable
test_state = np.random.rand(1,N_inputs)
test_state2 = np.copy(test_state)
test_state2[0,N_ego_states:N_ego_states+20] = test_state[0,N_inputs-20:N_inputs]
test_state2[0,N_inputs-20:N_inputs] = test_state[0,N_ego_states:N_ego_states+20]
t1 = nn.forward_pass(test_state)
t2 = nn.forward_pass(test_state2)
assert( (t1[0]== t2[0]).any() )
assert( (t1[1]== t2[1]).any() )


# #Test intermediate layers (working, just uncomment corresponding lines in neural_net.py)
# test_state_0 = np.zeros([1,N_inputs])
# conv_net1_out = nn.conv_net1_out.predict(test_state_0)
#
# test_state_ = copy.deepcopy(test_state_0)
# test_state_[0,0:N_ego_states] = np.ones([1,N_ego_states])
# test_state_[0,N_ego_states:N_ego_states+4] = np.array([[0.3, 0.2, 0.3, -1.0]])
# conv_net1_out1 = nn.conv_net1_out.predict(test_state_)
# test_state_[0,N_ego_states+4:N_ego_states+8] = np.array([[0.3, 0.2, 0.3, -1.0]])
# conv_net1_out2 = nn.conv_net1_out.predict(test_state_)
# assert( (conv_net1_out1[0,0] == conv_net1_out1[0,1]).any )
#
# conv_net2_out = nn.conv_net2_out.predict(test_state_)
# pool_out = nn.pool_out.predict(test_state_)
# assert( (conv_net2_out[0,0] == pool_out).any )
#
# test_state_2 = copy.deepcopy(test_state_)
# test_state_2[0,N_ego_states+4:N_ego_states+8] = np.array([[0.6, -0.2, 0.1, 0.8]])
# conv_net1_out_2 = nn.conv_net1_out.predict(test_state_2)
# conv_net2_out_2 = nn.conv_net2_out.predict(test_state_2)
# pool_out_2 = nn.pool_out.predict(test_state_2)
# assert( (np.maximum(conv_net2_out_2[0,0],conv_net2_out_2[0,1]) == pool_out_2).any )
#
# merged_out = nn.merged_out.predict(test_state_2)




start_time = time.time()
# Training
for i in range(0, 100):
    nn.add_samples_to_memory(state, train_dist, train_val)  # Should be state, mcts_dist, actual_value
    for i in range(0, 10):
        nn.update_network()
print(time.time() - start_time)

start_time = time.time()
for i in range(0,1000):
    nn.model.train_on_batch(state[0:32], [train_dist[0:32], train_val[0:32]])
print(time.time()-start_time)


#Save/load
dist_act1, est_val1 = nn.forward_pass(state)

nn.save_network("../Logs/testSave2")
nn2 = NeuralNetwork(N_inputs=nn.N_inputs,N_outputs=nn.N_outputs, replay_memory_max_size=nn.rm.max_size, training_start=nn.rm.training_start, log_path='../Logs/tmp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
nn2.load_network("../Logs/testSave2")

dist_act2, est_val2 = nn2.forward_pass(state)

assert(np.all(est_val1 == est_val2))
assert(np.all(dist_act1 == dist_act2))


#Test batch norm
dist_act_single, _ = nn.forward_pass(np.expand_dims(state[0],0))
assert(np.allclose(dist_act_single[0],dist_act1[0]))

print("Tests passed")
