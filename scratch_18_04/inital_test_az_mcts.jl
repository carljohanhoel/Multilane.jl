# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))
#include("../src/Multilane.jl")   #ZZZ This may be needed...

using Revise #To allow recompiling of modules withhout restarting julia

using Multilane
using MCTS
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


@everywhere using Missings
@everywhere using Multilane
@everywhere using POMDPToolbox


##
DEBUG = true #Debugging is also controlled from debug.jl

#Solver parameters
# @show N = 1000
@show n_iters = 1000 #10000   #C 1000
@show max_time = Inf
@show max_depth = 20 #60   #C 20
@show c_uct = 2.0   #C 5.0
@show k_state = 4.0 #0.2, #C 4.0,
@show alpha_state = 1/8.0 #0.0, #C 1/8.0,
# @show val = SimpleSolver()
alldata = DataFrame()


scenario = "continuous_driving"
# problem = "forced_lane_changes"


## Problem definition
if scenario == "continuous_driving"
    cor = 0.75

    #Reward
    lambda = 0.0
    lane_change_cost = 0.0

    nb_lanes = 3
    lane_length = 300.
    nb_cars = 20


    initSteps = 1000

    v_des = 25.0

    rmodel = SpeedReward(v_des=v_des, lane_change_cost=lane_change_cost, lambda=lambda)
elseif scenario == "forced_lane_changes"
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
pp = PhysicalParam(nb_lanes, lane_length=lane_length)
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

problem = mdp    #Choose which problem to work with
# problem = pomdp

## Solver definition
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
                 tree_in_info = DEBUG
                )


n_iter = 1000
depth = 20 #ZZZ not used?
c_puct = 10.
# replay_memory_max_size = 150
# training_start = 50
# training_steps = 1000
# save_freq = 200
# eval_freq = 200
# eval_eps = 3
replay_memory_max_size = 10000
training_start = 1000
training_steps = 1000000
save_freq = 1000
eval_freq = 2000
eval_eps = 10
rng = MersenneTwister(13)

some_state = initial_state(problem, initSteps=0)
n_s = length(MCTS.convert_state(some_state,mdp))
n_a = n_actions(problem)
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
log_name = length(ARGS)>0 ? ARGS[1] : ""
log_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"*Dates.format(Dates.now(), "yymmdd_HHMMSS_")*log_name
estimator = NNEstimator(rng, estimator_path, log_path, n_s, n_a, replay_memory_max_size, training_start)

azs = AZSolver(n_iterations=n_iter, depth=depth, exploration_constant=c_puct,
               k_state=3.,
               tree_in_info=true,
               alpha_state=0.2,
               enable_action_pw=false,
               check_repeat_state=false,
               rng=rng,
               estimate_value=estimator,
               init_P=estimator,
               noise_dirichlet = 4,
               noise_eps = 0.25
               )

solvers = Dict{String, Solver}(
    "baseline" => SingleBehaviorSolver(dpws, Multilane.NORMAL),
    "omniscient" => dpws,
    # "omniscient-x10" => dpws_x10,
    "mlmpc" => MLMPCSolver(dpws),
    "meanmpc" => MeanMPCSolver(dpws),)


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

method = "omniscient"
# method = "mlmpc" #Does not work with mdp
# solver = solvers[method]
solver = azs

sim_problem = deepcopy(problem)
sim_problem.throw=true


## Run simulations

# N = 25
# for i in 1:N
i = 1
rng_seed = i+40000
rng = MersenneTwister(rng_seed)
is = initial_state(sim_problem, rng, initSteps=initSteps)   #Init random state by simulating 200 steps with standard IDM model
# is = MLState(0.0, 0.0, CarState[CarState(pp.lane_length/2, 1, pp.v_med, 0.0, Multilane.NORMAL, 1),
#                                 CarState(pp.lane_length/2+20, 1, pp.v_med, 0.0, Multilane.TIMID, 1),
#                                 CarState(pp.lane_length/2, 2, pp.v_med, 0.0, Multilane.TIMID, 1)])
is = set_ego_behavior(is, ego_acc)
write_to_png(visualize(sim_problem,is,0),"./Figs/state_at_t0_i"*string(i)*".png")
#ZZZ Line below is temp, just to start with simple initial state
# is = Multilane.MLState(0.0, 0.0, Multilane.CarState[Multilane.CarState(50.0, 2.0, 30.0, 0.0, Multilane.IDMMOBILBehavior([1.4, 2.0, 1.5, 35.0, 2.0, 4.0], [0.6, 2.0, 0.1], 1), 1)], Nullable{Any}())
ips = MLPhysicalState(is)

metadata = Dict(:rng_seed=>rng_seed, #Not used now
                :lambda=>lambda,
                :solver=>solver,
                :dt=>pp.dt,
                :cor=>cor
           )
# hr = HistoryRecorder(max_steps=20, rng=rng, capture_exception=false, show_progress=true)
# hr_ref = HistoryRecorder(max_steps=20, rng=deepcopy(rng), capture_exception=false, show_progress=true)

hr = HistoryRecorder(max_steps=100, rng=rng, capture_exception=false, show_progress=false)

policy = solve(solver,sim_problem)
trainer = Trainer(rng=rng, training_steps=training_steps, save_freq=save_freq, eval_freq=eval_freq, eval_eps=eval_eps, show_progress=true, log_dir=log_path)
train(trainer, hr, mdp, policy)


##
