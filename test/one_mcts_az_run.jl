using Revise

push!(LOAD_PATH,joinpath("./src"))
using Multilane

using MCTS
using POMDPs
using POMDPModels
using POMDPToolbox
using Base.Test
using D3Trees

using Missings
using DataFrames
using CSV

#For viz
using AutoViz
using Reel
using ProgressMeter
using AutomotiveDrivingModels
using ImageView
using Images

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
@show c_puct = 2.0   #C 5.0
@show k_state = 4.0 #0.2, #C 4.0,
@show alpha_state = 1/8.0 #0.0, #C 1/8.0,
# @show val = SimpleSolver()
alldata = DataFrame()

## Problem definition
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


behaviors = standard_uniform(correlation=cor)   #Sets max/min values of IDM and MOBIL and how they are correlated.
pp = PhysicalParam(nb_lanes, lane_length=lane_length)
dmodel = NoCrashIDMMOBILModel(nb_cars, pp,   #First argument is number of cars
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=30000.0, #1000.0, #ZZZ Remember that the rollout policy must fit within this distance (Not for exit lane scenario)
                              vel_sigma = 0.5,   #0.0   #Standard deviation of speed of inserted cars
                              semantic_actions = true
                             )
mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)   #Third argument is discount factor
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)   #Fifth argument semantic action space

problem = mdp    #Choose which problem to work with
# problem = pomdp

## Solver definition
rollout_problem = deepcopy(problem)
rollout_problem.dmodel.semantic_actions = false
rollout_problem.dmodel.max_dist = Inf
rollout_behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 0.1), 1)
rollout_policy = Multilane.DeterministicBehaviorPolicy(rollout_problem, rollout_behavior, false)


rng = MersenneTwister(3)

n_s = 20   #ZZZ Parameterize
n_a = length(actions(problem))
estimator_path = "/home/cj/2018/Stanford/Code/Multilane.jl/src/nn_estimator"
estimator = NNEstimator(rng, estimator_path, n_s, n_a)

solver = AZSolver(n_iterations=n_iters, depth=max_depth, exploration_constant=c_puct,
                  k_state=3.,
                  tree_in_info=true,
                  alpha_state=0.2,
                  enable_action_pw=false,
                  check_repeat_state=false,
                  rng=rng,
                  estimate_value=estimator,
                  init_P=estimator
                  )


##
v_des = 25.0
ego_acc = ACCBehavior(ACCParam(v_des), 1)

sim_problem = deepcopy(problem)

i = 1
rng_seed = i+40000
rng = MersenneTwister(rng_seed)
is = initial_state(sim_problem, rng, initSteps=initSteps)   #Init random state by simulating 200 steps with standard IDM model
is = set_ego_behavior!(is, ego_acc)
write_to_png(visualize(sim_problem,is,0),"./Figs/state_at_t0_i"*string(i)*".png")
ips = MLPhysicalState(is)


policy = deepcopy(solve(solver, sim_problem))

a, ai = action_info(policy, is)
inchromium(D3Tree(ai[:tree],init_expand=1))

# hr = HistoryRecorder(max_steps=10, rng=rng, capture_exception=false, show_progress=true)
# hr_ref = HistoryRecorder(max_steps=10, rng=deepcopy(rng), capture_exception=false, show_progress=true)
#
# planner = deepcopy(solve(solver, sim_problem))
# srand(planner, rng_seed+60000)   #Sets rng seed of planner
# hist = simulate(hr, sim_problem, planner, is)
# hist_ref = simulate(hr_ref, sim_problem, rollout_policy, is)


##############
#DPW reference
solver_dpw = DPWSolver(depth=max_depth,
                 n_iterations=n_iters,
                 max_time=max_time,
                 exploration_constant=c_puct,
                 k_state=k_state,
                 alpha_state=alpha_state,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(rollout_policy),
                 # estimate_value=val
                 tree_in_info = DEBUG
                )


policy_dpw = solve(solver_dpw, sim_problem)

a_dpw, ai_dpw = action_info(policy_dpw, is)

inchromium(D3Tree(ai_dpw[:tree],init_expand=1))
