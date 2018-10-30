# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

# parallel_version = true   #Test code in parallel mode
parallel_version = false

# simple_run = true
simple_run = false

tree_in_info = true

sample_to_load = "6081"
network_to_load = "181024_155209_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample"
logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"


include("simulation_setup.jl")


##
#Possibly add loop here, loop over process
process = 7
i=process-2
rng_base_seed = 15
rng_seed = 100*(i-1)+rng_base_seed
rng = MersenneTwister(rng_seed)
s_initial = initial_state(sim_problem, rng, initSteps=initSteps)
s_initial = set_ego_behavior(s_initial, ego_acc)
o_initial = MLObs(s_initial, problem.dmodel.phys_param.sensor_range, problem.dmodel.phys_param.obs_behaviors)

##
if sim_problem isa POMDP
    if solver isa MLMPCSolver
        updater = make_updater(cor, sim_problem, rng_seed)
        planner = deepcopy(solve(solver, sim_problem))
        srand(planner, rng_seed+60000)   #Sets rng seed of planner
        planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
        hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
    else
        updater = LimitedRangeUpdater()
        planner = deepcopy(solve(solver, sim_problem))
        srand(planner, rng_seed+60000)   #Sets rng seed of planner
        planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
        hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
    end
else
    planner = deepcopy(solve(solver, sim_problem))
    # planner.solver.n_iterations = 1
    srand(planner, rng_seed+60000)   #Sets rng seed of planner
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
frames = Frames(MIME("image/png"), fps=10/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/rerunAZ_process_"*string(process)*"_"*network_to_load*"_"*sample_to_load*".ogv"
write(gifname, frames)
