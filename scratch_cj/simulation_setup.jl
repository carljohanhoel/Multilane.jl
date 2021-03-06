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
if isdefined(:network_to_load)
    include(logs_path*network_to_load*"/code/scratch_cj/parameters.jl")
else
    include("parameters.jl")
end

if isdefined(:depth)
    max_depth = depth
end

behaviors = standard_uniform(correlation=cor)   #Sets max/min values of IDM and MOBIL and how they are correlated.

############# TEST ##############
behaviors.max_mobil = MOBILParam(0.0, behaviors.max_mobil[2], behaviors.max_mobil[3])   #Sets politeness factor to 0 for all vehicles.
#################################

pp = PhysicalParam(nb_lanes, lane_length=lane_length, sensor_range=sensor_range, obs_behaviors=obs_behaviors)
dmodel = NoCrashIDMMOBILModel(nb_cars, pp,   #First argument is number of cars
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=false,
                              max_dist=30000.0, #1000.0, #ZZZ Remember that the rollout policy must fit within this distance (Not for exit lane scenario)
                              vel_sigma = 0.5,   #0.0   #Standard deviation of speed of inserted cars
                              init_state_steps = initSteps,
                              semantic_actions = true
                             )
if scenario=="exit_lane"
    dmodel.max_dist = exit_distance
end
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


# Solver definition
# For both cases
rollout_problem = deepcopy(problem)
rollout_problem.dmodel.semantic_actions = false
rollout_problem.dmodel.max_dist = Inf
rollout_behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 0.1), 1)
rollout_policy = Multilane.DeterministicBehaviorPolicy(rollout_problem, rollout_behavior, false)
if scenario == "exit_lane"
    # rollout_policy = SimpleSolver()
    rollout_beh = IDMLaneSeekingSolver(rollout_behavior)
    rollout_policy = solve(rollout_beh,mdp)
end




# rng_estimator=MersenneTwister(rng_seed+1) #Not used
rng_evaluator=MersenneTwister(rng_seed+2)
rng_solver=MersenneTwister(rng_seed+4)     #This should correspond to what's in trainer eval eps to be able to compare
rng_history=MersenneTwister(rng_seed+3)    #This should correspond to what's in trainer eval eps to be able to compare
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
   estimator = NNEstimator(estimator_path, log_path, n_s, n_a, v_min, v_max, replay_memory_max_size, training_start)
end

if isdefined(:network_to_load)
    load_network(estimator,logs_path*network_to_load*"/"*sample_to_load)
end


azs = AZSolver(n_iterations=n_iter, depth=max_depth, exploration_constant=c_puct,
               k_state=k_state,
               tree_in_info=tree_in_info,
               alpha_state=alpha_state,
               tau=tau,
               enable_action_pw=false,
               check_repeat_state=false,
               rng=rng_solver,
               estimate_value=estimator,
               init_P=estimator,
               noise_dirichlet=noise_dirichlet,
               noise_eps=noise_eps
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

policy = solve(solver,sim_problem)
