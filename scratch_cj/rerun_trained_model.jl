# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

# parallel_version = true   #Test code in parallel mode
parallel_version = false

# simple_run = true
simple_run = false

tree_in_info = true

sample_to_load = "7031"
# network_to_load = "181016_140842_driving_Change_pen_0p01_Loss_weights_1_10_Cpuct_0p1_Remove_10_samples_Only_z_target_No_vehicles"
network_to_load = "181119_180615_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Added_set_V_set_T_ego_state"

logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"


include("simulation_setup.jl")

# hr.max_steps = 10

##
#Possibly add loop here, loop over process
process = 5
i=process-2

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


#Visualization
#Set time t used for showing tree. Use video to find interesting situations.
t = 0.0
step = convert(Int, t / pp.dt) + 1
write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step]),"./Figs/state_at_t"*string(t)*"_"*network_to_load*"_"*sample_to_load*".png")
print(hist.action_hist[step])
inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))
# inchromium(D3Tree(hist.ainfo_hist[step][:tree],hist.state_hist[step],init_expand=1))   #For MCTS (not DPW)


#Produce video
frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/rerunAZ_process_"*string(process)*"_"*network_to_load*"_"*sample_to_load*".ogv"
write(gifname, frames)
