# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

@show scenario = "continuous_driving"
# @show scenario = "exit_lane"

parallel_version = false
simple_run = false
tree_in_info = true


sample_to_load = "15011"
# sample_to_load = "10022"

# network_to_load = "181126_154437_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim"
network_to_load = "181126_155336_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim_Weights_1_10"

# network_to_load = "181130_160730_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim"
# network_to_load = "181203_174746_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim_Terminal_state_Est_v0"

logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"

include("simulation_setup.jl")

episode_length = 20
hr.max_steps = episode_length

i=1
s_initial = initial_double_overtaking_state(sim_problem)

## IDM/MOBIL
rollout_problem = deepcopy(sim_problem)
rollout_problem.dmodel.semantic_actions = false
rollout_problem.dmodel.max_dist = Inf
rollout_behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 0.1), 1)
rollout_policy = Multilane.DeterministicBehaviorPolicy(rollout_problem, rollout_behavior, false)

hr_ref = HistoryRecorder(max_steps=episode_length, rng=deepcopy(rng_history), capture_exception=false, show_progress=true)
hist_ref = simulate(hr_ref, sim_problem, rollout_policy, s_initial)

## MCTS
c_uct = c_puct
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
                 tree_in_info = tree_in_info
                )
policy_dpw = solve(solver,sim_problem)

hr_dpw = HistoryRecorder(max_steps=episode_length, rng=deepcopy(rng_history), capture_exception=false, show_progress=true)
hist_dpw = simulate(hr_dpw, sim_problem, policy_dpw, s_initial)


## AZ
planner = deepcopy(solve(solver, sim_problem))
planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
hist = simulate(hr, sim_problem, planner, s_initial)




@show sum(hist.reward_hist)
@show hist.state_hist[end].x


##Visualization
if !ispath(logs_path*network_to_load*"/SpecialCase")
   mkdir(logs_path*network_to_load*"/SpecialCase")
end
t = 0.0
step = convert(Int, t / pp.dt) + 1
write_to_png(visualize(sim_problem,hist_ref.state_hist[step],hist_ref.reward_hist[step], high_quality=true),logs_path*network_to_load*"/SpecialCase/"*"ref_state_at_t_"*string(t)*".png")
write_to_png(visualize(sim_problem,hist_dpw.state_hist[step],hist_dpw.reward_hist[step], high_quality=true),logs_path*network_to_load*"/SpecialCase/"*"dpw_state_at_t_"*string(t)*".png")
write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step], high_quality=true),logs_path*network_to_load*"/SpecialCase/"*"sample_"*sample_to_load*"_state_at_t_"*string(t)*".png")

t = 14.25
step = convert(Int, t / pp.dt) + 1
write_to_png(visualize(sim_problem,hist_ref.state_hist[step],hist_ref.reward_hist[step], high_quality=true),logs_path*network_to_load*"/SpecialCase/"*"ref_state_at_t_"*string(t)*".png")
write_to_png(visualize(sim_problem,hist_dpw.state_hist[step],hist_dpw.reward_hist[step], high_quality=true),logs_path*network_to_load*"/SpecialCase/"*"dpw_state_at_t_"*string(t)*".png")
write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step], high_quality=true),logs_path*network_to_load*"/SpecialCase/"*"sample_"*sample_to_load*"_state_at_t_"*string(t)*".png")


##Produce video
frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist_ref, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = logs_path*network_to_load*"/SpecialCase/ref.ogv"
write(gifname, frames)

frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist_dpw, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = logs_path*network_to_load*"/SpecialCase/dpw.ogv"
write(gifname, frames)

frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = logs_path*network_to_load*"/SpecialCase/"*"sample_"*sample_to_load*".ogv"
write(gifname, frames)


#Vis options
include("visualize_value_action.jl")


#Save result to file
log = MCTS.create_eval_log(sim_problem, hist, 1, parse(Int,sample_to_load))
open(logs_path*network_to_load*"/SpecialCase/"*"results.txt","a") do f
    writedlm(f, log, " ")
end


# end
