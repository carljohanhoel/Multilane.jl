include("../src/Multilane.jl")
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

DEBUG = true

# @show N = 1000
@show n_iters = 1000 #10000   #C 1000
@show max_time = Inf
@show max_depth = 20 #60   #C 20
@show c_uct = 5.0 #1.0   #C 5.0
@show k_state = 4.0 #0.2, #C 4.0,
@show alpha_state = 1/8.0 #0.0, #C 1/8.0,
@show val = SimpleSolver()
alldata = DataFrame()

dpws = DPWSolver(depth=max_depth,
                 n_iterations=n_iters,
                 max_time=max_time,
                 exploration_constant=c_uct,
                 k_state=k_state,
                 alpha_state=alpha_state,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(val),
                 # estimate_value=val
                 tree_in_info = DEBUG
                )

# mcts_solver = MCTSSolver(n_iterations=n_iters,
#                          depth=max_depth,
#                          exploration_constant=5.0,
#                          estimate_value=RolloutEstimator(val),
#                          enable_tree_vis = DEBUG
#                         )

dpws_x10 = deepcopy(dpws)
dpws_x10.n_iterations *= 10

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

cor = 0.75
lambda = 1.0

@show cor
@show lambda

behaviors = standard_uniform(correlation=cor)
pp = PhysicalParam(4, lane_length=100.0)
dmodel = NoCrashIDMMOBILModel(10, pp,
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=2000.0 #1000.0,
                              # vel_sigma = 0.5   #0.0   #Standard deviation of speed of inserted cars
                             )
rmodel = SuccessReward(lambda=lambda)
mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

problems = Dict{String,Any}("omniscient"=>mdp, "mlmpc"=>pomdp)

method = "omniscient"
# method = "mlmpc"
# solver = mcts_solver
# problem = mdp
solver = solvers[method]
problem = problems[method]
sim_problem = deepcopy(problem)
sim_problem.throw=true


# for i in 1:N
i = 1
rng_seed = i+40000
rng = MersenneTwister(rng_seed)
initial_state = initial_state(sim_problem, rng)
ips = MLPhysicalState(initial_state)

metadata = Dict(:rng_seed=>rng_seed, #Not used now
                :lambda=>lambda,
                :solver=>solver,
                :dt=>pp.dt,
                :cor=>cor
           )
hr = HistoryRecorder(max_steps=100, rng=rng, capture_exception=false)


if sim_problem isa POMDP
    updater = make_updater(cor, sim_problem, rng_seed)
    planner = deepcopy(solve(solver, sim_problem))
    srand(planner, rng_seed+60000)   #Sets rng seed of planner
    hist = simulate(hr, sim_problem, planner, updater, ips, initial_state)
    # hist = simulate(hr, sim_problem, planner, updater, initial_belief, initial_state)
else
    planner = deepcopy(solve(solver, sim_problem))
    srand(planner, rng_seed+60000)   #Sets rng seed of planner
    hist = simulate(hr, sim_problem, planner, initial_state)
end


#Visualization
#Set time t used for showing tree. Use video to find interesting situations.
t = 11.25
step = convert(Int, t / pp.dt) + 1
write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step]),"/home/cj/2018/Multilane/Figs/state_at_t.png")
print(hist.action_hist[step])
inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))
# inchromium(D3Tree(hist.ainfo_hist[step][:tree],hist.state_hist[step],init_expand=1))   #For MCTS (not DPW)

#Produce video
frames = Frames(MIME("image/png"), fps=1/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/testMCTS.ogv"
write(gifname, frames)


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
