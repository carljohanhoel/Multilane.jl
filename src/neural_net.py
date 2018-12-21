from __future__ import print_function

import itertools
import joblib
import numpy as np
import random
import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   #Deactivates GPU support (to be tested...)

from keras.models import Model, load_model
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Input, Reshape, concatenate, Lambda, Conv1D, MaxPooling1D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.optimizers import sgd
from keras import regularizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1 #ZZZ Should be correlated with network size
set_session(tf.Session(config=config))

class ReplayMemory():
    def __init__(self,N_inputs,N_outputs,replay_memory_max_size, training_start):
        self.states = [np.zeros(N_inputs)] * replay_memory_max_size  # Dynamic
        self.dists = [np.zeros(N_outputs)] * replay_memory_max_size  # Dynamic
        self.vals = [np.zeros(1)] * replay_memory_max_size  # Dynamic
        self.max_size = replay_memory_max_size
        self.write_idx = 0  # Dynamic
        self.size = 0  # Dynamic
        self.training_start = training_start

class NeuralNetwork:
    def __init__(self, N_inputs, N_outputs, replay_memory_max_size, training_start, log_path="./", batch_size=32, c=0.0001, loss_weights=[1, 100], lr=1e-2, debug=False):
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        self.batch_size = batch_size
        self.batch_no = 0   #Dynamic
        self.log_path = log_path

        self.rm = ReplayMemory(N_inputs, N_outputs, replay_memory_max_size, training_start)

        self.c = c
        self.loss_weights = loss_weights
        self.lr = lr

        self.model_name = time.strftime('G%y%m%dT%H%M%S')
        print(self.model_name)

        if N_inputs == 3:   #GridWorld test case
            self.create_simple()
        else:
            self.create_convnet()

        self.debug = debug


    def create_simple(self):
        N_inputs = self.N_inputs
        N_outputs = self.N_outputs
        N_size = 32

        state = Input(shape=(N_inputs,))
        joint_net = Dense(N_size, activation='relu', kernel_regularizer=regularizers.l2(self.c))(state)
        joint_net = Dense(N_size, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)

        dist = Dense(N_size, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)
        dist = Dense(N_outputs, activation='softmax', name='probabilities', kernel_regularizer=regularizers.l2(self.c))(dist)

        val = Dense(N_size, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)
        val = Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(self.c))(val)

        self.model = Model(state, [dist, val])
        optimizer = sgd(lr=self.lr, decay=0, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'mean_squared_error'], loss_weights=self.loss_weights)
        self.model.summary()

        self.tf_callback = TensorBoard(log_dir=self.log_path, histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                       write_grads=False,
                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                       embeddings_metadata=None)
        self.tf_callback.set_model(self.model)

    def create_convnet(self):
        N_inputs = self.N_inputs
        N_outputs = self.N_outputs

        N_vehicles = 20
        N_inputs_per_vehicle = 4
        N_inputs_ego_vehicle = N_inputs - N_vehicles*N_inputs_per_vehicle
        N_conv_filters = 32

        state = Input(shape=(N_inputs,))

        state_ego = Lambda(lambda state : state[:,:N_inputs_ego_vehicle])(state)
        state_others = Lambda(lambda state: state[:, N_inputs_ego_vehicle:])(state)

        state_others_reshaped = Reshape((N_vehicles*N_inputs_per_vehicle,1,),input_shape=(N_vehicles*N_inputs_per_vehicle,))(state_others)
        conv_net = Conv1D(N_conv_filters, N_inputs_per_vehicle, strides=N_inputs_per_vehicle, use_bias=False, kernel_regularizer=regularizers.l2(self.c))(state_others_reshaped)
        conv_net = BatchNormalization()(conv_net)
        conv_net = Activation(activation='relu')(conv_net)
        conv_net2 = Conv1D(N_conv_filters, 1, strides=1, use_bias=False, kernel_regularizer=regularizers.l2(self.c))(conv_net)
        conv_net2 = BatchNormalization()(conv_net2)
        conv_net2 = Activation(activation='relu')(conv_net2)
        #pool = MaxPooling1D(pool_size=1, strides=N_conv_filters)(conv_net2) #This was a previous bug. Pool size should be N_vehicles.
        pool = MaxPooling1D(pool_size=N_vehicles)(conv_net2)
        conv_net_out = Reshape((N_conv_filters,),input_shape=(1,N_conv_filters,))(pool)

        merged = concatenate([state_ego, conv_net_out])

        joint_net = Dense(64, use_bias=False, kernel_regularizer=regularizers.l2(self.c))(merged)
        joint_net = BatchNormalization()(joint_net)
        joint_net = Activation(activation='relu')(joint_net)

        # joint_net = Dense(64, use_bias=False, kernel_regularizer=regularizers.l2(self.c))(joint_net)
        # joint_net = BatchNormalization()(joint_net)
        # joint_net = Activation(activation='relu')(joint_net)

        dist = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)
        dist = Dense(N_outputs, activation='softmax', name='probabilities', kernel_regularizer=regularizers.l2(self.c))(dist)

        val = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)
        val = Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(self.c))(val)

        self.model = Model(state, [dist, val])
        optimizer = sgd(lr=self.lr, decay=0, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'mean_squared_error'], loss_weights=self.loss_weights)
        self.model.summary()

        self.tf_callback = TensorBoard(log_dir=self.log_path, histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                       write_grads=False,
                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                       embeddings_metadata=None)
        self.tf_callback.set_model(self.model)

        #self.intermediate_layers = {}
        #for i in range(0,20):
        #    self.intermediate_layers[i] = Model(inputs=self.model.input, outputs=self.model.get_layer(index=i).output)

        # #Just for debugging, to get intermediate outputs
        # self.conv_net1_out = Model(state, [conv_net])
        # self.conv_net2_out = Model(state, [conv_net2])
        # self.pool_out = Model(state, [pool])
        # self.merged_out = Model(state, [merged])

    def add_samples_to_memory(self, states, dists, vals):
        idx = self.rm.write_idx
        dists += np.nextafter(0,1) #Add eps to avoid problems with log(0.0)=-inf in network update
        ns = len(states)
        if idx + ns <= self.rm.max_size:
            self.rm.states[idx:idx+ns] = states.tolist()
            self.rm.dists[idx:idx+ns] = dists.tolist()
            self.rm.vals[idx:idx+ns] = vals.tolist()
            self.rm.write_idx += ns
        else:
            self.rm.states[idx:] = states.tolist()[0:self.rm.max_size-idx]
            self.rm.dists[idx:] = dists.tolist()[0:self.rm.max_size - idx]
            self.rm.vals[idx:] = vals.tolist()[0:self.rm.max_size - idx]
            self.rm.states[0:ns-(self.rm.max_size-idx)] = states.tolist()[self.rm.max_size-idx:]
            self.rm.dists[0:ns - (self.rm.max_size - idx)] = dists.tolist()[self.rm.max_size - idx:]
            self.rm.vals[0:ns - (self.rm.max_size - idx)] = vals.tolist()[self.rm.max_size - idx:]
            self.rm.write_idx = ns-(self.rm.max_size-idx)

        self.rm.size = min(self.rm.max_size,self.rm.size+ns)
        assert(self.rm.max_size == len(self.rm.states)), "replay memory grew out of bounds"

        if self.debug: print("Memory size: "+str(self.rm.size)+", write idx: "+str(self.rm.write_idx))


    def update_network(self):
        if self.rm.size >= self.rm.training_start:
            idx = np.random.choice(np.arange(self.rm.size), self.batch_size)
            archive_states = np.array(self.rm.states)[idx]
            archive_dists = np.array(self.rm.dists)[idx]
            archive_vals = np.array(self.rm.vals)[idx]
        else:
            return #Do nothing until replay memory is bigger than training start

        # print(archive_vals)
        # _, v_tmp = self.model.predict(archive_states)
        # print(v_tmp)
        # print(np.sum((archive_vals-v_tmp)**2)/len(v_tmp))

        logs = self.model.train_on_batch(archive_states, [archive_dists, archive_vals])  # C Backprop

        if self.debug: print("Updates: "+str(self.batch_no))

        #Tensorboard log
        nn = ['loss', 'probabilities_loss','value_loss', 'absolute value error', '"ideal" cross entropy', 'actual - ideal crossentropy']
        # print(logs)
        # print(archive_dists)
        # print(len(archive_dists))
        # print(self.rm.size)
        # print(idx)
        # print( -np.sum(archive_dists * np.log(archive_dists)) )
        # print( np.log(archive_dists) )
        # print( len(archive_dists) )

        data = logs
        data.append(np.sqrt(logs[2]))
        data.append(-np.sum(archive_dists * np.log(archive_dists)) / len(archive_dists))
        data.append(data[1]-data[4])
        self.write_log(self.tf_callback, nn, data, self.batch_no)
        self.batch_no+=1


    def forward_pass(self, states):
        return self.model.predict(states)

    def save_network(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save('%s.weights.h5' % (filename,))
        joblib.dump([self.rm, self.batch_no],
                    '%s.archive.joblib' % (filename,), compress=5)

    def load_network(self, filename):
        self.model = load_model('%s.weights.h5' % (filename,))

        pos_fname = '%s.archive.joblib' % (filename,)
        try:
            [self.rm, self.batch_no] = joblib.load(pos_fname)
        except:
            print('Warning: Cannot load memory archive %s' % (pos_fname,))

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()