# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

parallel_version = false
simple_run = false
tree_in_info = false

# sample_to_load = "6081"
# network_to_load = "181024_155209_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample"
# logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"

##
#Code for loading saved networks, running them with only the NN policy and printing the result to a file
logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"
network_to_load = "181024_155209_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample"
eval_samples = []
all_files = readdir(logs_path*network_to_load)
old_results = ""
if in("prior_policy_result.txt",all_files)
    f = open(logs_path*network_to_load*"/prior_policy_result.txt")
    old_results = read(f, String)
    close(f)
end
for file_name in all_files
    if startswith(file_name, ['1','2','3','4','5','6','7','8','9'])
        sample = file_name[1:findfirst(isequal('.'),file_name)-1]
        if !in(sample,eval_samples)
            if size(findfirst("1 "*sample,old_results))[1] == 0
                push!(eval_samples,sample)
            end
        end
    end
end

##

sample_to_load = eval_samples[1] #Just needed for the first load in simulation_setup.jl
include("simulation_setup.jl")

##
for sample_to_load in eval_samples
    println("Running sample "*sample_to_load)
    load_network(estimator,logs_path*network_to_load*"/"*sample_to_load)

    ##
    for process in 3:22
        #process = 7
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
            planner.solver.n_iterations = 1
            srand(planner, rng_seed+60000)   #Sets rng seed of planner
            planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
            hist = simulate(hr, sim_problem, planner, s_initial)
        end

        @show sum(hist.reward_hist)
        @show hist.state_hist[end].x


        open(logs_path*network_to_load*"/prior_policy_result.txt","a") do f
            writedlm(f, [[i, sample_to_load, sum(hist.reward_hist), hist.state_hist[end].x]], " ")
        end
    end

end

#
#
# #Visualization
# #Set time t used for showing tree. Use video to find interesting situations.
# t = 0.0
# step = convert(Int, t / pp.dt) + 1
# write_to_png(visualize(sim_problem,hist.state_hist[step],hist.reward_hist[step]),"./Figs/state_at_t"*string(t)*"_"*network_to_load*"_"*sample_to_load*".png")
# print(hist.action_hist[step])
# inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))
# # inchromium(D3Tree(hist.ainfo_hist[step][:tree],hist.state_hist[step],init_expand=1))   #For MCTS (not DPW)
#
#
# #Produce video
# frames = Frames(MIME("image/png"), fps=10/pp.dt)
# @showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
#     push!(frames, visualize(problem, s, r))
# end
# gifname = "./Figs/rerunAZ_process_"*string(process)*"_"*network_to_load*"_"*sample_to_load*".ogv"
# write(gifname, frames)
