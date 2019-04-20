# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

@show scenario = "continuous_driving"
# @show scenario = "exit_lane"

# parallel_version = true   #Test code in parallel mode
parallel_version = false

# simple_run = true
simple_run = false

tree_in_info = false
# tree_in_info = true




# Continuous case
# network_to_load = "181119_180615_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Added_set_V_set_T_ego_state"
# network_to_load = "181126_154437_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim"
# network_to_load = "181126_155336_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim_Weights_1_10"
network_to_load = "181221_153735_driving_Cpuct_0p1_Bigger_net_New_action_space"

# sample_to_load = "381"
# sample_to_load = "1141"
# sample_to_load = "2091"
# sample_to_load = "3041"
# sample_to_load = "4181"
# sample_to_load = "5131"
# sample_to_load = "6081"
# sample_to_load = "7031"
# sample_to_load = "8171"
# sample_to_load = "9121"
# sample_to_load = "10071"
# sample_to_load = "11021"
# sample_to_load = "12161"
# sample_to_load = "13111"

sample_to_load = "13111"




# Exit case
# network_to_load = "181130_160730_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim"
# network_to_load = "181203_174746_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim_Terminal_state_Est_v0"
# network_to_load = "181215_121952_driving_exit_lane_Cpuct_0p5_Dpw_0p3_Big_replay_Truck_dim_Terminal_state_Est_v0_R_plus_19"
# network_to_load = "190128_142353_driving_exit_lane_Cpuct_0p1_Bigger_net_New_action_space_No_batchnorm"

# sample_to_load = "1035"
# sample_to_load = "2054"
# sample_to_load = "3056"
# sample_to_load = "4028"
# sample_to_load = "5050"
# sample_to_load = "6041"

# sample_to_load = "15038"


logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"


include("simulation_setup.jl")

solver.n_iterations = 10000


# hr.max_steps = 10

hist = []      #Needs to be defines so can us include visualize_value_action
planner = []   #Needs to be defines so can us include visualize_value_action
process = []   #Needs to be defines so can us include visualize_value_action
##
#Possibly add loop here, loop over process
for process in 93:102
# for process in 72:102
# process = 5
i=process-2

@show i

# Reset rng:s (not necessary if not running a loop)
rng_evaluator_copy=MersenneTwister(Int(rng_evaluator.seed[1])+100*(i-1))
rng_history_copy=MersenneTwister(Int(rng_history.seed[1])+100*(i-1))
hr.rng = rng_history_copy

s_initial = initial_state(sim_problem, rng_evaluator_copy)
s_initial = set_ego_behavior(s_initial, ego_acc)
o_initial = MLObs(s_initial, problem.dmodel.phys_param.sensor_range, problem.dmodel.phys_param.obs_behaviors)

##
if sim_problem isa POMDP
    if solver isa MLMPCSolver
        updater = make_updater(cor, sim_problem, rng_seed+2+100*(i-1))
        planner = deepcopy(solve(solver, sim_problem))
        srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
        planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
        hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
    else
        updater = LimitedRangeUpdater()
        planner = deepcopy(solve(solver, sim_problem))
        srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
        planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
        hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
    end
else
    planner = deepcopy(solve(solver, sim_problem))
    # planner.solver.n_iterations = 1
    srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
    planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
    hist = simulate(hr, sim_problem, planner, s_initial)
end

@show sum(hist.reward_hist)
@show hist.state_hist[end].x

#
# #Visualization
# #Set time t used for showing tree. Use video to find interesting situations.
# if !ispath(logs_path*network_to_load*"/Reruns")
#    mkdir(logs_path*network_to_load*"/Reruns")
# end
# t = 0.0
# step = convert(Int, t / pp.dt) + 1
# # write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step]),"./Figs/state_at_t"*string(t)*"_"*network_to_load*"_"*sample_to_load*".png")
# write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step]),logs_path*network_to_load*"/Reruns/"*"sample_"*sample_to_load*"_process_"*string(process)*"_state_at_t_"*string(t)*"_"*network_to_load*".png")
# # print(hist.action_hist[step])
# # inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))
#
#
# #Produce video
# frames = Frames(MIME("image/png"), fps=3/pp.dt)
# @showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
#     push!(frames, visualize(problem, s, r))
# end
# gifname = logs_path*network_to_load*"/Reruns/"*"sample_"*sample_to_load*"_process_"*string(process)*"_rerun_"*network_to_load*".ogv"
# write(gifname, frames)
#
#
# #Vis options
# include("visualize_value_action.jl")


#Save result to file
log = MCTS.create_eval_log(sim_problem, hist, process, parse(Int,sample_to_load))
log[1][2] = solver.n_iterations
# open(logs_path*network_to_load*"/Reruns/"*"evalResults2_searches_"*sample_to_load*".txt","a") do f
open(logs_path*network_to_load*"/Reruns/"*"evalResults2_searches_"*sample_to_load*".txt","a") do f
    writedlm(f, log, " ")
end


end
