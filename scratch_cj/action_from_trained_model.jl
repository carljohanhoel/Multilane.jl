# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

# parallel_version = true   #Test code in parallel mode
parallel_version = false

# simple_run = true
simple_run = false


if parallel_version
   n_workers = 20
   if simple_run
      n_workers = 4
   end
   addprocs(n_workers+1)
   @everywhere using Multilane
   @everywhere using MCTS
else
   n_workers = 1
   using Multilane
   using MCTS
end

using POMDPToolbox
using POMDPs
# using POMCP
using Missings
using DataFrames
using CSV
using POMCPOW

#For viz
using AutoViz
using Reel
using ProgressMeter
using AutomotiveDrivingModels
using ImageView
using Images
#include("../src/visualization.jl")

#For tree viz
using D3Trees
@everywhere using D3Trees

##



@show scenario = "continuous_driving"
@show problem_type = "mdp"

## Problem definition
if scenario == "continuous_driving"
    cor = 0.75

    #Reward
    lambda = 0.0
    lane_change_cost = 0.01 #1.0

    nb_lanes = 4
    lane_length = 600.
    nb_cars = 20
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

@show cor
@show lambda

behaviors = standard_uniform(correlation=cor)   #Sets max/min values of IDM and MOBIL and how they are correlated.

############# TEST ##############
behaviors.max_mobil = MOBILParam(0.0, behaviors.max_mobil[2], behaviors.max_mobil[3])   #Sets politeness factor to 0 for all vehicles.
#################################

pp = PhysicalParam(nb_lanes, lane_length=lane_length, sensor_range=sensor_range, obs_behaviors=obs_behaviors)
dmodel = NoCrashIDMMOBILModel(nb_cars, pp,   #First argument is number of cars
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=30000.0, #1000.0, #ZZZ Remember that the rollout policy must fit within this distance (Not for exit lane scenario)
                              vel_sigma = 0.5,   #0.0   #Standard deviation of speed of inserted cars
                              init_state_steps = initSteps,
                              semantic_actions = true
                             )
mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)   #Third argument is discount factor
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)   #Fifth argument semantic action space
pomdp_lr = NoCrashPOMDP_lr{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

if problem_type == "mdp"
    problem = mdp
elseif problem_type == "pomdp"
    if pomdp.dmodel.phys_param.obs_behaviors
        problem = pomdp_lr
    else
        problem = pomdp
    end
end

##DPW solver parameters (not used for AZ!)!!!!!!!!!!!
@show n_iters = 1000 #10000   #C 1000
max_time = Inf
max_depth = 20 #60   #C 20
@show c_uct = 2.0   #C 5.0
k_state = 4.0 #0.2, #C 4.0,
alpha_state = 1/8.0 #0.0, #C 1/8.0,
# @show val = SimpleSolver()
alldata = DataFrame()

# Solver definition
if scenario == "continuous_driving"
    rollout_problem = deepcopy(problem)
    rollout_problem.dmodel.semantic_actions = false
    rollout_problem.dmodel.max_dist = Inf
    rollout_behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 0.1), 1)
    rollout_policy = Multilane.DeterministicBehaviorPolicy(rollout_problem, rollout_behavior, false)
elseif scenario == "forced_lane_changes"
    rollout_policy = SimpleSolver()
end

dpws = DPWSolver(depth=max_depth,
                 n_iterations=n_iters,
                 max_time=max_time,
                 exploration_constant=c_uct,
                 k_state=k_state,
                 alpha_state=alpha_state,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(rollout_policy),
                 # estimate_value=val
                 tree_in_info = false
                )


@show n_iter = 2000
depth = 20 #ZZZ not used
@show c_puct = 0.1 #1. #5.
@show tau = 1.1
@show stash_factor = 1.5
@show noise_dirichlet = 1.0

if simple_run
    episode_length = 20
    n_iter = 20
    replay_memory_max_size = 200
    training_start = 100
    training_steps = Int(ceil(1000/n_workers))*1000
    n_network_updates_per_sample = 1
    # save_freq = Int(ceil(100/n_workers))
    # eval_freq = Int(ceil(100/n_workers))
    # eval_eps = Int(ceil(2/n_workers))
    save_freq = 1*episode_length
    eval_freq = 1*episode_length
    eval_eps = 1
else
    episode_length = 200
    replay_memory_max_size = 10000 #ZZZ This should probably be increased since each episode is 200 long. But keep it short to begin with, to see if it learns something.
    training_start = 5000
    training_steps = Int(ceil(100000000/n_workers))
    n_network_updates_per_sample = 1
    # save_freq = Int(ceil(5000/n_workers))
    # eval_freq = Int(ceil(5000/n_workers))
    # eval_eps = Int(ceil(5/n_workers))
    save_freq = 1*episode_length
    eval_freq = 1*episode_length
    eval_eps = 1
end
rng = MersenneTwister(13)

rng_seed = 13
rng_estimator=MersenneTwister(rng_seed+1)
rng_evaluator=MersenneTwister(rng_seed+2)
rng_solver=MersenneTwister(rng_seed+3)
rng_history=MersenneTwister(rng_seed+4)
rng_trainer=MersenneTwister(rng_seed+5)

some_state = initial_state(problem, initSteps=0)
n_s = length(MCTS.convert_state(some_state,problem))
n_a = n_actions(problem)
v_min, v_max = max_min_cum_reward(problem)
v_max += 0.1*(v_max-v_min) #To make it easier for sigmoid to reach max or min value
v_min -= 0.1*(v_max-v_min)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/neural_net"
log_name = length(ARGS)>0 ? ARGS[1] : ""
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS_")*log_name

if parallel_version
   #Start queue on process 2
   @spawnat 2 run_queue(NetworkQueue(estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start, false),cmd_queue,res_queue)
   estimator = NNEstimatorParallel(v_min, v_max)
   sleep(3) #Wait for queue to be set up before continuing
   clear_queue()
else
   estimator = NNEstimator(rng_estimator, estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)
end

# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180727_013135_driving_20_workers_eval_every_episode/1001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180727_180659_driving_20_workers_eval_every_episode_fixed/61201")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180914_114955_driving_20_workers_Eval_5_eps_Lane_penalty_start_Two_steps_per_change/5001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180918_142056_driving_20_workers_Eval_5_eps_New_scenarios/8001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180921_160842_driving_20_workers_Eval_5_eps_New_scenarios_1puct_Q_init_v0/8001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180924_175301_driving_20_workers_Eval_5_eps_New_scenarios_1puct_Q_init_v0_No_branching_1_0/1001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180925_131535_driving_20_workers_Eval_5_eps_New_scenarios_1puct_Q_init_v0_No_branching_Change_penalty_0p01/1001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180926_174518_driving_20_workers_Eval_5_eps_New_scenarios_1puct_Q_init_v0_No_branching_Change_penalty_0p01/1001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180927_152157_driving_20_worker_etc_Change_pen_0p01_Loss_weights_1_1000/5001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/181002_121926_driving_20_worker_etc_Change_pen_0p01_Loss_weights_1_1000_fixed/8001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/181002_143554_driving_20_worker_etc_Change_pen_0p01_Loss_weights_1_1000_Only_z_target/8001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/180926_174518_driving_20_workers_Eval_5_eps_New_scenarios_1puct_Q_init_v0_No_branching_Change_penalty_0p01/22001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/181009_182754_driving_20_worker_etc_Change_pen_0p01_Loss_weights_1_10_No_vehicles/3001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/181009_183215_driving_20_worker_etc_Change_pen_0p01_Loss_weights_1_10_One_vehicle/3001")
# load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/181012_150851_driving_20_worker_etc_Change_pen_0p01_Loss_weights_1_10_Cpuct_0p1/3001")
load_network(estimator,"/home/cj/2018/Stanford/Code/Multilane.jl/Logs/181012_151511_driving_20_worker_etc_Change_pen_0p01_Loss_weights_1_10_No_vehicles_Cpuct_0p1/3001")


# estimator.debug_with_uniform_nn_output = true

azs = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=c_puct,
               k_state=1., #3.,
               tree_in_info=true,
               alpha_state=0., #0.2,
               tau=tau,
               enable_action_pw=false,
               check_repeat_state=false,
               rng=rng_solver,
               estimate_value=estimator,
               init_P=estimator,
               noise_dirichlet = noise_dirichlet,
               noise_eps = 0.25
               )


function make_updater(cor, problem, rng_seed)
    wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
    if cor >= 1.0
        return AggressivenessUpdater(problem, 2000, 0.05, 0.1, wup, MersenneTwister(rng_seed+50000))
    else
        return BehaviorParticleUpdater(problem, 5000, 0.05, 0.2, wup, MersenneTwister(rng_seed+50000))
    end
end

pow_updater(up::AggressivenessUpdater) = AggressivenessPOWFilter(up.params)
pow_updater(up::BehaviorParticleUpdater) = BehaviorPOWFilter(up.params)

v_des = 25.0
ego_acc = ACCBehavior(ACCParam(v_des), 1)

## Choice of solver
if problem isa POMDP
    if problem.dmodel.phys_param.obs_behaviors
        solver = LimitedRangeSolver(azs) #limited sensor range
    else
        solver = MLMPCSolver(azs) #limited sensor range and estimated behaviors
    end
else
    solver = azs #omniscient
end

sim_problem = deepcopy(problem)
sim_problem.throw=true

hr = HistoryRecorder(max_steps=episode_length, rng=rng_history, capture_exception=false, show_progress=false)

if sim_problem isa POMDP
    if solver isa MLMPCSolver
        updater = make_updater(cor, problem, rng_seed)
        # policy = solve(solver,sim_problem)
        # srand(policy, rng_seed+60000)
        # trainer = Trainer(rng=rng_trainer, rng_eval=rng_evaluator, training_steps=training_steps, n_network_updates_per_episode=n_network_updates_per_episode, save_freq=save_freq, eval_freq=eval_freq, eval_eps=eval_eps, fix_eval_eps=true, show_progress=true, log_dir=log_path)
        # train(trainer, hr, problem, policy, updater)
    else
        updater = LimitedRangeUpdater()
        # policy = solve(solver,sim_problem)
        # srand(policy, rng_seed+60000)
        # trainer = Trainer(rng=rng_trainer, rng_eval=rng_evaluator, training_steps=training_steps, n_network_updates_per_episode=n_network_updates_per_episode, save_freq=save_freq, eval_freq=eval_freq, eval_eps=eval_eps, fix_eval_eps=true, show_progress=true, log_dir=log_path)
        # train(trainer, hr, problem, policy, updater)
    end
else
    # policy = solve(solver,sim_problem)
    # srand(policy, rng_seed+60000)
    # trainer = Trainer(rng=rng_trainer, rng_eval=rng_evaluator, training_steps=training_steps, n_network_updates_per_episode=n_network_updates_per_episode, save_freq=save_freq, eval_freq=eval_freq, eval_eps=eval_eps, fix_eval_eps=true, show_progress=true, log_dir=log_path)
    # train(trainer, hr, problem, policy)
end

policy = solve(solver,sim_problem)   #Not used
srand(policy, rng_seed+5)   #Not used


##

#Load state from log
# saved_data = JLD.load("./Logs/states_run_i.jld")
saved_data = JLD.load("./Logs/saved_states_no_vehicles.jld")
rec_states = saved_data["state_hist"]
t = 119.25
step = convert(Int, t / pp.dt) + 1
test_state = rec_states[step]

#Change acc time state
acc = test_state.cars[1].behavior.p_idm
acc2 = [value for (idx,value) in enumerate(acc)]
# acc2[4] = 24.0
# acc2[3] = 0.5
test_state.cars[1] = CarState(test_state.cars[1].x,test_state.cars[1].y,test_state.cars[1].vel,test_state.cars[1].lane_change, ACCBehavior(ACCParam(acc2),1), test_state.cars[1].id)
# test_state.cars[1] = CarState(test_state.cars[1].x,4.0,test_state.cars[1].vel,test_state.cars[1].lane_change, ACCBehavior(ACCParam(acc2),1), test_state.cars[1].id)

#Run MCTS
policy.training_phase = false #Remove Dirichlet noise on prior probabilities
a, ai = action_info(policy, test_state)

write_to_png(visualize(sim_problem,rec_states[step],0.0),"./Figs/state_at_t.png")
inchromium(D3Tree(ai[:tree],init_expand=1))

##Plot value for different lanes
s_ego_veh = test_state.cars[1]
v0 = zeros(7,1)
p0_vec = zeros(7,5)
for i in 1:7
    y = i*0.5 + 0.5
    s_ego_veh = CarState(s_ego_veh.x, y, s_ego_veh.vel, (mod(y,1.0)>0)*1.0, s_ego_veh.behavior, s_ego_veh.id)
    test_state.cars[1] = s_ego_veh
    v0[i] = estimate_value(policy.solved_estimate, policy.mdp, test_state, policy.solver.depth)[1]

    allowed_actions = actions(policy.mdp, test_state)   #This is handled a bit weird to make it compatible with existing structure of Multilane.jl
    all_actions = actions(policy.mdp)
    if length(allowed_actions) == length(all_actions)
        allowed_actions_vec = ones(Float64, 1, length(all_actions))
    else
        allowed_actions_vec = zeros(Float64, 1, length(all_actions))
        for idx in collect(allowed_actions.acceptable)
            allowed_actions_vec[idx] = 1.0
        end
    end
    p0_vec[i,:] = MCTS.init_P(policy.solver.init_P, policy.mdp, test_state, allowed_actions_vec)
end
write_to_png(visualize_with_nn(sim_problem,test_state,0.0,v0,p0_vec),"./Figs/estimated_values.png")