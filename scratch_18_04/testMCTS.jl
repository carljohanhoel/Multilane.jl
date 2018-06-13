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
    lane_length = 600.
    nb_cars = 20

    initSteps = 200

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
solver = solvers[method]

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
hr = HistoryRecorder(max_steps=200, rng=rng, capture_exception=false, show_progress=true)
hr_ref = HistoryRecorder(max_steps=200, rng=deepcopy(rng), capture_exception=false, show_progress=true)

##

if sim_problem isa POMDP
    updater = make_updater(cor, sim_problem, rng_seed)
    planner = deepcopy(solve(solver, sim_problem))
    srand(planner, rng_seed+60000)   #Sets rng seed of planner
    hist = simulate(hr, sim_problem, planner, updater, ips, is)
    hist_ref = simulate(hr_ref, sim_problem, rollout_policy, updater, ips, is)
    # hist = simulate(hr, sim_problem, planner, updater, initial_belief, initial_state)
else
    planner = deepcopy(solve(solver, sim_problem))
    srand(planner, rng_seed+60000)   #Sets rng seed of planner
    hist = simulate(hr, sim_problem, planner, is)
    hist_ref = simulate(hr_ref, sim_problem, rollout_policy, is)
end

@show sum(hist.reward_hist)
@show sum(hist_ref.reward_hist)
@show hist.state_hist[end].x
@show hist_ref.state_hist[end].x


#Visualization
#Set time t used for showing tree. Use video to find interesting situations.
# t = 4.5
# step = convert(Int, t / pp.dt) + 1
# write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step]),"./Figs/state_at_t.png")
# print(hist.action_hist[step])
# inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))
# # inchromium(D3Tree(hist.ainfo_hist[step][:tree],hist.state_hist[step],init_expand=1))   #For MCTS (not DPW)


#Produce video
frames = Frames(MIME("image/png"), fps=10/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/testMCTS_i"*string(i)*".ogv"
write(gifname, frames)

#Reference model
frames = Frames(MIME("image/png"), fps=10/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist_ref, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/testMCTS_i"*string(i)*"_ref.ogv"
write(gifname, frames)

# end

# For visualizing rollouts, not used for now. See make_video for more details
# tree = get(hist.ainfo_hist[1], :tree, nothing)
# rollouts = make_rollouts(planner, tree)
# nwr = NodeWithRollouts(POWTreeObsNode(tree, 1), rollouts)
# push!(frames, visualize(pomdp, is, r, tree=nwr))


#----------


#
# success = 100.0*sum(data[:terminal].=="lane")/N
# brakes = 100.0*sum(data[:nb_brakes].>=1)/N
# @printf("%% reaching:%5.1f; %% braking:%5.1f\n", success, brakes)
#
# @show extrema(data[:distance])
# @show mean(data[:mean_iterations])
# @show mean(data[:mean_search_time])
# @show mean(data[:reward])
# if minimum(data[:min_speed]) < 15.0
#     @show minimum(data[:min_speed])
# end
#
# if isempty(alldata)
#     alldata = data
# else
#     alldata = vcat(alldata, data)
# end
#
# datestring = Dates.format(now(), "E_d_u_HH_MM")
# filename = joinpath("/tmp", "uncor_gap_checkpoint_"*datestring*".csv")
# println("Writing data to $filename")
# CSV.write(filename, alldata)
# # end
# #     end
# # end
#
# # @show alldata
#
# datestring = Dates.format(now(), "E_d_u_HH_MM")
# filename = Pkg.dir("Multilane", "data", "uncor_gap_"*datestring*".csv")
# println("Writing data to $filename")
# CSV.write(filename, alldata)
