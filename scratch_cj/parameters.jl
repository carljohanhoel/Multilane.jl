@show scenario = "continuous_driving"
@show problem_type = "mdp"

## Problem definition
if scenario == "continuous_driving"
    cor = 0.75

    #Reward
    lambda = 0.0  #Penalty for making other vehicles brake hard
    lane_change_cost = 0.03 #0.01 #0.1 #0.01 #1.0 #*0.1

    nb_lanes = 4
    lane_length = 600.
    nb_cars = 20     #Number of vehicles, including ego vehicle
    sensor_range = 200.   #Remember that this also affects the IDM/MOBIL model
    @show obs_behaviors = false   #Estimate or observe other vehicles' behaviors in pomdp setting

    initSteps = 150   #To create initial random state

    v_des = 25.0

    rmodel = SpeedReward(v_des=v_des, lane_change_cost=lane_change_cost, lambda=lambda)
elseif scenario == "forced_lane_changes" #ZZZ deprecated
    cor = 0.75
    lambda = 1.0

    nb_lanes = 4
    lane_length = 100.
    nb_cars = 10

    initSteps = 200

    v_des = 35.0
    rmodel = SuccessReward(lambda=lambda)
end



# Parameters AZ solver
@show n_iter = 2000
depth = 20 #ZZZ not used
@show c_puct = 0.1 #1. #5.
@show k_state = 1.0 #3.0
@show alpha_state = 0.0 #0.2
@show tau = 1.1
@show stash_factor = 1.5
@show noise_dirichlet = 1.0
@show noise_eps = 0.25

if simple_run
    episode_length = 20
    n_iter = 20
    replay_memory_max_size = 200
    training_start = 100
    training_steps = Int(ceil(1000/n_workers))*1000
    n_network_updates_per_sample = 3
    remove_end_samples = 2
    # save_freq = Int(ceil(100/n_workers))
    # eval_freq = Int(ceil(100/n_workers))
    # eval_eps = Int(ceil(2/n_workers))
    save_freq = 1*episode_length
    eval_freq = 1*episode_length
    eval_eps = 1
    save_evaluation_history = true
else
    episode_length = 200
    replay_memory_max_size = 20000 #ZZZ This should probably be increased since each episode is 200 long. But keep it short to begin with, to see if it learns something.
    training_start = 5000
    training_steps = Int(ceil(100000000/n_workers))
    n_network_updates_per_sample = 3
    remove_end_samples = 10
    # save_freq = Int(ceil(5000/n_workers))
    # eval_freq = Int(ceil(5000/n_workers))
    # eval_eps = Int(ceil(5/n_workers))
    save_freq = 5*episode_length
    eval_freq = 5*episode_length
    eval_eps = 1
    save_evaluation_history = true
end




rng_seed = 13
