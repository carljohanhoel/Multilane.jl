push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

using Multilane
using MCTS

using POMDPToolbox
using POMDPs
using POMCPOW

#For viz
using AutoViz
using Reel
using ProgressMeter
using AutomotiveDrivingModels
using ImageView
using Images

#For tree viz
using D3Trees

@show scenario = "continuous_driving"
# @show scenario = "exit_lane"

# simple_run = true
simple_run = false
n_workers = 1

include("parameters.jl")
c_uct = c_puct

##
DEBUG = true #Debugging is also controlled from debug.jl
start_time = Dates.format(Dates.now(), "yymmdd_HHMMSS_")


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

idle_problem = deepcopy(problem)
idle_problem.dmodel.semantic_actions = false
idle_problem.dmodel.max_dist = Inf
idle_behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 1000000), 1) #Very high acc gain treshold means no lane changes
idle_policy = Multilane.DeterministicBehaviorPolicy(idle_problem, idle_behavior, false)

dpws = DPWSolver(depth=max_depth,
                 n_iterations=n_iter,
                 max_time=Inf,
                 exploration_constant=c_uct,
                 k_state=k_state,
                 alpha_state=alpha_state,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(rollout_policy),
                 # estimate_value=val
                 tree_in_info = DEBUG
                )


rng_evaluator=MersenneTwister(rng_seed+2)
rng_solver=MersenneTwister(rng_seed+4)     #This should correspond to what's in trainer eval eps to be able to compare
rng_history=MersenneTwister(rng_seed+3)    #This should correspond to what's in trainer eval eps to be able to compare
rng_trainer=MersenneTwister(rng_seed+5)



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
        solver = LimitedRangeSolver(dpws) #limited sensor range
    else
        solver = MLMPCSolver(dpws) #limited sensor range and estimated behaviors
    end
else
    solver = dpws #omniscient
end

sim_problem = deepcopy(problem)
sim_problem.throw=true
# sim_problem.throw=false

policy = solve(solver,sim_problem)
srand(policy, rng_seed+5)



## Run simulations
# N = 20
for i in 88:88
# i=3

# Reset rng:s
rng_evaluator_copy=MersenneTwister(Int(rng_evaluator.seed[1])+100*(i-1))
rng_history_copy=MersenneTwister(Int(rng_history.seed[1])+100*(i-1))

s_initial = initial_state(sim_problem, rng_evaluator_copy)
s_initial = set_ego_behavior(s_initial, ego_acc)
o_initial = MLObs(s_initial, problem.dmodel.phys_param.sensor_range, problem.dmodel.phys_param.obs_behaviors)

write_to_png(visualize(sim_problem,s_initial,0),"./Figs/ref_model_state_at_t0_i"*string(i)*".png")

metadata = Dict(:rng_seed=>rng_seed, #Not used now
                :lambda=>lambda,
                :solver=>solver,
                :dt=>pp.dt,
                :cor=>cor
           )
hr = HistoryRecorder(max_steps=episode_length, rng=deepcopy(rng_history_copy), capture_exception=false, show_progress=true)
hr_ref = HistoryRecorder(max_steps=episode_length, rng=deepcopy(rng_history_copy), capture_exception=false, show_progress=true)
hr_idle = HistoryRecorder(max_steps=episode_length, rng=deepcopy(rng_history_copy), capture_exception=false, show_progress=true)



##

if sim_problem isa POMDP
    if solver isa MLMPCSolver
        updater = make_updater(cor, sim_problem, rng_seed+2+100*(i-1))
        planner = deepcopy(policy)
        srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
        hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
        hist_ref = simulate(hr_ref, sim_problem, rollout_policy, updater, o_initial, s_initial)
        hist_idle = simulate(hr_idle, sim_problem, idle_policy, updater, o_initial, s_initial)
    else
        updater = LimitedRangeUpdater()
        planner = deepcopy(policy)
        srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
        hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
        hist_ref = simulate(hr_ref, sim_problem, rollout_policy, updater, o_initial, s_initial)
        hist_idle = simulate(hr_idle, sim_problem, idle_policy, updater, o_initial, s_initial)
    end
else
    planner = deepcopy(policy)
    srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
    hist = simulate(hr, sim_problem, planner, s_initial)
    hist_ref = simulate(hr_ref, sim_problem, rollout_policy, s_initial)
    hist_idle = simulate(hr_idle, sim_problem, idle_policy, s_initial)
end

@show sum(hist.reward_hist)
@show sum(hist_ref.reward_hist)
@show sum(hist_idle.reward_hist)
@show hist.state_hist[end].x
@show hist_ref.state_hist[end].x
@show hist_idle.state_hist[end].x
if scenario == "exit_lane"
    @show hist.state_hist[end].t
    println(hist.state_hist[end].cars[1].y == 1.0)
    @show hist_ref.state_hist[end].t
    println(hist_ref.state_hist[end].cars[1].y == 1.0)
    @show hist_idle.state_hist[end].t
    println(hist_idle.state_hist[end].cars[1].y == 1.0)
end

#Action distribution
action_dist_dpw = zeros(Int64,5)
as = NoCrashSemanticActionSpace(sim_problem).actions
for action in hist.action_hist
    action_dist_dpw[find(as .== action)[1]] += 1
end

action_dist_ref = zeros(Int64,5)
for action in hist_ref.action_hist
    if action.lane_change == 0.
        action_dist_ref[1] += 1
    elseif action.lane_change == -1.
        action_dist_ref[4] += 1
    elseif action.lane_change == +1.
        action_dist_ref[5] += 1
    end
end


if scenario == "continuous_driving"
    logname = "dpwAndRefAndIdleModelsDistance"
elseif scenario == "exit_lane"
    logname = "exit_dpwAndRefAndIdleModelsDistance"
end
open("./Logs/dpwAndRefAndIdleModelsDistance_"*scenario*"_"*start_time*".txt","a") do f
    # writedlm(f, [[i, sum(hist_ref.reward_hist), sum(hist_idle.reward_hist), hist_ref.state_hist[end].x, hist_idle.state_hist[end].x]], " ")
    if scenario == "continuous_driving"
        writedlm(f, [[i, sum(hist.reward_hist), sum(hist_ref.reward_hist), sum(hist_idle.reward_hist), hist.state_hist[end].x, hist_ref.state_hist[end].x, hist_idle.state_hist[end].x, action_dist_dpw[1], action_dist_dpw[2], action_dist_dpw[3], action_dist_dpw[4], action_dist_dpw[5], action_dist_ref[1], action_dist_ref[4], action_dist_ref[5] ]], " ")
    elseif scenario == "exit_lane"
        writedlm(f, [[i, sum(hist.reward_hist), sum(hist_ref.reward_hist), sum(hist_idle.reward_hist), hist.state_hist[end].x, hist_ref.state_hist[end].x, hist_idle.state_hist[end].x,
                         hist.state_hist[end].t, hist_ref.state_hist[end].t, hist_idle.state_hist[end].t, hist.state_hist[end].t-(hist.state_hist[end].x-dmodel.max_dist)/hist.state_hist[end].cars[1].vel, hist_ref.state_hist[end].t-(hist_ref.state_hist[end].x-dmodel.max_dist)/hist_ref.state_hist[end].cars[1].vel, hist_idle.state_hist[end].t-(hist_idle.state_hist[end].x-dmodel.max_dist)/hist_idle.state_hist[end].cars[1].vel,
                         hist.state_hist[end].cars[1].y, hist_ref.state_hist[end].cars[1].y, hist_idle.state_hist[end].cars[1].y, hist.state_hist[end].cars[1].y==1.0 *1, hist_ref.state_hist[end].cars[1].y==1.0 *1, hist_idle.state_hist[end].cars[1].y==1.0 *1 , action_dist_dpw[2], action_dist_dpw[3], action_dist_dpw[4], action_dist_dpw[5], action_dist_ref[1], action_dist_ref[4], action_dist_ref[5] ]], " ")
    end
end

# open("./Logs/refModelResults2_.txt","a") do f
#     writedlm(f, [[sum(hist_ref.reward_hist)]], " ")
#     writedlm(f, [[hist_ref.state_hist[end].x]], " ")
# end

# end
#
# #Visualization
# #Set time t used for showing tree. Use video to find interesting situations.
# t = 4.5
# step = convert(Int, t / pp.dt) + 1
# write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step]),"./Figs/state_at_t.png")
# print(hist.action_hist[step])
# inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))
# # inchromium(D3Tree(hist.ainfo_hist[step][:tree],hist.state_hist[step],init_expand=1))   #For MCTS (not DPW)

#
#Produce video
frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/"*start_time*"_"*scenario*"_"*string(i)*"_dpw"*".ogv"
write(gifname, frames)


#Reference model
frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist_ref, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/"*start_time*"_"*scenario*"_"*string(i)*"_refModel"*".ogv"
write(gifname, frames)
#
#Idle model
frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist_idle, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/"*start_time*"_"*scenario*"_"*string(i)*"_idleModel"*".ogv"
write(gifname, frames)



end







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
