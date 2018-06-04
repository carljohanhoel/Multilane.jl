import numpy as np
from multiprocessing import Process, Queue
import sys

from time import sleep #ZZZZZ tmp debug




class ModelServer(Process):
    def __init__(self, cmd_queue, res_queues, N_states, N_actions, replay_memory_max_size, training_start, log_path, load_snapshot=None):
        super(ModelServer, self).__init__()
        self.cmd_queue = cmd_queue
        self.res_queues = res_queues
        self.load_snapshot = load_snapshot

        self.N_states = N_states
        self.N_actions = N_actions
        self.replay_memory_max_size = replay_memory_max_size
        self.training_start = training_start
        self.log_path = log_path

    def run(self):
        try:
            from neural_net import AGZeroModel
            net = AGZeroModel(self.N_states, self.N_actions, self.replay_memory_max_size, self.training_start,
                               self.log_path)   #C Set network architecture
            net.create_simple()   #C Create network
            if self.load_snapshot is not None:
                net.load(self.load_snapshot)

            class PredictStash(object):
                """ prediction batcher """
                def __init__(self, trigger, res_queues):
                    self.stash = []
                    self.trigger = trigger  # XXX must not be higher than #workers
                    self.res_queues = res_queues

                def add(self, kind, X_pos, ri):
                    self.stash.append((kind, X_pos, ri))
                    if len(self.stash) >= self.trigger:
                        self.process()

                def process(self):
                    if not self.stash:
                        return
                    dist, res = net.predict([s[1] for s in self.stash])     #C Bundles up forward passes to speed up
                    for d, r, s in zip(dist, res, self.stash):
                        kind, _, ri = s
                        self.res_queues[ri].put(d if kind == 0 else r) #C Return distribution of 'predict_distribution' is asked for, otherwise return value, when 'predict_winrate' is asked for
                    self.stash = []

            stash = PredictStash(1, self.res_queues)
            update_counter = 0

            while True:
                cmd, args, ri = self.cmd_queue.get() #C Waits for queue to be non-empty, then gets first  element
                # print("got new item") # ZZZZZZZZZZZZ remove debug
                # print("Queue size ")
                # print(self.cmd_queue.qsize())
                if cmd == 'stash_size':
                    stash.process()
                    stash.trigger = args['stash_size']
                elif cmd == 'update_network':
                    stash.process()   #C This empties the queue of forward passes before updating the NN
                    print('\rUpdate %d...' % (update_counter,), end='')
                    sys.stdout.flush()
                    update_counter += 1
                    # print("updating")
                    net.update_network(**args)
                    self.res_queues[ri].put("update_done") #Dummy value, just used to force program to wait for the result
                elif cmd == 'predict_distribution':
                    stash.add(0, args['state'], ri)   #C Add forward pass request to the queue. ri identifies which process that requested it.
                elif cmd == 'predict_value':
                    stash.add(1, args['state'], ri)   #C Add forward pass request to the queue. ri identifies which process that requested it.
                elif cmd == 'model_name':
                    self.res_queues[ri].put(net.model_name)
                elif cmd == 'save':
                    stash.process()
                    net.save(args['snapshot_id'])
        except:
            import traceback
            traceback.print_exc()



class GoModel(object):
    def __init__(self, N_states, N_actions, replay_memory_max_size, training_start, log_path, load_snapshot=None):
        self.cmd_queue = Queue()   #C Queue of commands that we want to calculate
        self.res_queues = [Queue() for i in range(128)]   #C ZZZ Hm, think the 128 here is a max of number of processes. When reporting result of a requested forward pass, it is put at the index corresponding to the process that requested it.
        self.server = ModelServer(self.cmd_queue, self.res_queues, N_states, N_actions, replay_memory_max_size, training_start, log_path, load_snapshot=load_snapshot)
        self.server.start()   #C Starts run function under ModelServer. This will then run the while loop and wait for things to be put in the queue.
        self.ri = 0  # id of process in case of multiple processes, to prevent mixups

    def stash_size(self, stash_size):
        self.cmd_queue.put(('stash_size', {'stash_size': stash_size}, self.ri))

    def update_network(self, states, dists, vals):   #C Training of NN
        # print("in queing, update network")
        print(self.ri)

        self.cmd_queue.put(('update_network', {'states': states, 'dists': dists, 'vals': vals}, self.ri))  # C Do the backprop step
        # print("put in queue")

        tmp = self.res_queues[self.ri].get() #To force waiting for result
        # print(tmp)

    def predict_distribution(self, state):   #Calculates prior distribution of a state from NN
        self.cmd_queue.put(('predict_distribution', {'state': state}, self.ri)) #C Add state to queue of evaluating NN
        return self.res_queues[self.ri].get()   #C Returns result when evaluation is done. Returns position corresponding to process ID.

    def predict_value(self, state):   #Calculate value of state from NN
        self.cmd_queue.put(('predict_value', {'state': state}, self.ri)) #C Add state to evaluation queue
        return self.res_queues[self.ri].get()   #C Return value when evaluation done

    def model_name(self):
        self.cmd_queue.put(('model_name', {}, self.ri))
        return self.res_queues[self.ri].get()

    def save(self, snapshot_id):
        self.cmd_queue.put(('save', {'snapshot_id': snapshot_id}, self.ri))

    def terminate(self):
        self.server.terminate()
