from __future__ import print_function

import itertools
import joblib
import numpy as np
import random
import time

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


############################
# AlphaGo Zero style network

class ResNet(object):
    def __init__(self, input_N=256, filter_N=256, n_stages=19,
                 kernel_width=3, kernel_height=3,
                 inpkern_width=3, inpkern_height=3):
        # number of filters and dimensions of the initial input kernel
        self.input_N = input_N
        self.inpkern_width = inpkern_width
        self.inpkern_height = inpkern_height
        # base number of filters and dimensions of the followup resnet kernels
        self.filter_N = filter_N
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.n_stages = n_stages

    def create(self, width, height, n_planes):
        bn_axis = 3
        inp = Input(shape=(width, height, n_planes))

        x = inp
        x = Conv2D(self.input_N, (self.inpkern_width, self.inpkern_height), padding='same', name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)

        for i in range(self.n_stages):
            x = self.identity_block(x, [self.filter_N, self.filter_N], stage=i+1, block='a')

        self.model = Model(inp, x)
        return self.model

    def identity_block(self, input_tensor, filters, stage, block):
        '''The identity_block is the block that has no conv layer at shortcut

        # Arguments
            input_tensor: input tensor
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        '''
        nb_filter1, nb_filter2 = filters
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = input_tensor
        x = Conv2D(nb_filter1, (self.kernel_width, self.kernel_height),
                          padding='same', name=conv_name_base + 'a')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (self.kernel_width, self.kernel_height),
                          padding='same', name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)

        x = add([x, input_tensor])
        return x


class AGZeroModel:
    def __init__(self, N_inputs, N_outputs, replay_memory_max_size, training_start, log_path="./", batch_size=32, c=0.00001, loss_weights=[1, 10], lr=1e-2):
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        self.batch_size = batch_size
        self.batch_no = 0
        self.log_path = log_path

        self.archive_fit_samples = 64 #deprecated(?)
        self.position_archive = [] #deprecated
        # self.replay_memory = []
        self.replay_memory = [None]*replay_memory_max_size
        self.replay_memory_max_size = replay_memory_max_size
        self.replay_memory_write_idx = 0
        self.replay_memory_size = 0
        self.training_start = training_start

        self.c = c
        self.loss_weights = loss_weights
        self.lr = lr

        self.model_name = time.strftime('G%y%m%dT%H%M%S')
        print(self.model_name)

    def create(self):
        bn_axis = 3

        N_outputs= self.N_outputs
        position = Input((N_outputs, N_outputs, 6))
        resnet = ResNet(n_stages=N_outputs)
        resnet.create(N_outputs, N_outputs, 6)
        x = resnet.model(position)

        dist = Conv2D(2, (1, 1))(x)
        dist = BatchNormalization(axis=bn_axis)(dist)
        dist = Activation('relu')(dist)
        dist = Flatten()(dist)
        dist = Dense(N_outputs * N_outputs + 1, activation='softmax', name='distribution')(dist)

        res = Conv2D(1, (1, 1))(x)
        res = BatchNormalization(axis=bn_axis)(res)
        res = Activation('relu')(res)
        res = Flatten()(res)
        res = Dense(256, activation='relu')(res)
        res = Dense(1, activation='sigmoid', name='result')(res)

        self.model = Model(position, [dist, res])
        self.model.compile(Adam(lr=self.lr), ['categorical_crossentropy', 'binary_crossentropy'])
        self.model.summary()

    def create_simple(self):
        N_inputs = self.N_inputs
        N_outputs = self.N_outputs

        state = Input(shape=(N_inputs,))
        joint_net = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.c))(state)
        joint_net = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)

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

    def create_convnet(self):
        N_inputs = self.N_inputs
        N_outputs = self.N_outputs

        N_vehicles = 20
        N_inputs_per_vehicle = 4
        N_conv_filters = 32

        state = Input(shape=(N_inputs,))

        state_ego = Lambda(lambda state : state[:,:2])(state)
        state_others = Lambda(lambda state: state[:, 2:])(state)

        state_others_reshaped = Reshape((N_vehicles*N_inputs_per_vehicle,1,),input_shape=(N_vehicles*N_inputs_per_vehicle,))(state_others)
        conv_net = Conv1D(N_conv_filters, N_inputs_per_vehicle, strides=N_inputs_per_vehicle, use_bias=False, kernel_regularizer=regularizers.l2(self.c))(state_others_reshaped)
        conv_net = BatchNormalization()(conv_net)
        conv_net = Activation(activation='relu')(conv_net)
        conv_net2 = Conv1D(N_conv_filters, 1, strides=1, use_bias=False, kernel_regularizer=regularizers.l2(self.c))(conv_net)
        conv_net2 = BatchNormalization()(conv_net2)
        conv_net2 = Activation(activation='relu')(conv_net2)
        pool = MaxPooling1D(pool_size=1, strides=N_conv_filters)(conv_net2)
        conv_net_out = Reshape((N_conv_filters,),input_shape=(1,N_conv_filters,))(pool)

        merged = concatenate([state_ego, conv_net_out])

        joint_net = Dense(64, use_bias=False, kernel_regularizer=regularizers.l2(self.c))(merged)
        joint_net = BatchNormalization()(joint_net)
        joint_net = Activation(activation='relu')(joint_net)

        dist = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)
        dist = Dense(N_outputs, activation='softmax', name='probabilities', kernel_regularizer=regularizers.l2(self.c))(dist)

        val = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(self.c))(joint_net)
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

    def fit_game(self, X_positions, result): #ZZZ not used, just kept for reference as of now
        X_posres = []
        for pos, dist in X_positions:
            X_posres.append((pos, dist, result))
            result = -result

        self.position_archive.extend(X_posres)

        if len(self.position_archive) >= self.archive_fit_samples:
            archive_samples = random.sample(self.position_archive, self.archive_fit_samples)
        else:
            # initial case
            archive_samples = self.position_archive
        # I'm going to some lengths to avoid the potentially overloaded + operator
        X_fit_samples = list(itertools.chain(X_posres, archive_samples)) #C Samples are picked in a strange way here. Adds the current samples to the archive.
        X_shuffled = random.sample(X_fit_samples, len(X_fit_samples))

        X, y_dist, y_res = [], [], []
        for pos, dist, res in X_shuffled:
            X.append(pos)
            y_dist.append(dist)
            y_res.append(float(res) / 2 + 0.5)
            if len(X) % self.batch_size == 0:
                self.model.train_on_batch(np.array(X), [np.array(y_dist), np.array(y_res)])   #C Backprop
                X, y_dist, y_res = [], [], []
        if len(X) > 0:
            self.model.train_on_batch(np.array(X), [np.array(y_dist), np.array(y_res)])   #C Backprop

    def add_samples_to_memory(self, states, dists, vals):
        assert not (states == None).any(), print("none state present\n" + str(states))
        assert not (dists == None).any(), print("none dist present\n" + str(dists))
        assert not (vals == None).any(), print("none val present\n" + str(vals))
        new_samples = []
        for i in range(0,len(states)):   #ZZZ This can be done faster
            new_samples.append([states[i],dists[i],vals[i]])
        idx = self.replay_memory_write_idx
        ns = len(new_samples)
        if idx + ns <= self.replay_memory_max_size:
            self.replay_memory[idx:idx+ns] = new_samples
            self.replay_memory_write_idx += ns
        else:
            self.replay_memory[idx:] = new_samples[0:self.replay_memory_max_size-idx]
            self.replay_memory[0:ns-(self.replay_memory_max_size-idx)] = new_samples[self.replay_memory_max_size-idx:]
            self.replay_memory_write_idx = ns-(self.replay_memory_max_size-idx)

        self.replay_memory_size = min(self.replay_memory_max_size, self.replay_memory_size + ns)
        assert (self.replay_memory_max_size == len(self.replay_memory)), "replay memory grew out of bounds"

    def update_network(self):
        if self.replay_memory_size >= self.training_start:
            if self.replay_memory_size == self.replay_memory_max_size:
                archive_samples = random.sample(self.replay_memory, self.batch_size)
            else:
                archive_samples = random.sample(self.replay_memory[0:self.replay_memory_write_idx-1], self.batch_size)
        else:
            return #Do nothing until replay memory is bigger than training start

        batch_states, batch_dists, batch_vals = [], [], []
        for state, dist, val in archive_samples:
            batch_states.append(state)
            batch_dists.append(dist)
            batch_vals.append(val)
        logs = self.model.train_on_batch(np.array(batch_states), [np.array(batch_dists), np.array(batch_vals)])   #C Backprop

        #Tensorboard log
        nn = ['loss', 'probabilities_loss','value_loss', 'absolute value error']
        data = logs
        data.append(np.sqrt(logs[2]))
        self.write_log(self.tf_callback, nn, data, self.batch_no)
        self.batch_no+=1

    def predict(self, states):
        dist, res = self.model.predict(states)
        return [dist, res]

    def save(self, snapshot_id):
        self.model.save('%s.weights.h5' % (snapshot_id,))
        joblib.dump(self.replay_memory, '%s.archive.joblib' % (snapshot_id,), compress=5)

    def load(self, snapshot_id):
        self.model = load_model('%s.weights.h5' % (snapshot_id,))
        # self.model.load_weights('%s.weights.h5' % (snapshot_id,))

        pos_fname = '%s.archive.joblib' % (snapshot_id,)
        try:
            self.replay_memory = joblib.load(pos_fname)
        except:
            print('Warning: Cannot load position archive %s' % (pos_fname,))

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()