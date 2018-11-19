# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

parallel_version = false
simple_run = false
tree_in_info = false

##
#Code for loading saved networks, running them with only the NN policy and printing the result to a file
logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"
network_to_load = "181116_155907_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_No_batchnorm"
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
    # for process in 5:5
        i=process-2

        # Reset rng:s
        rng_evaluator_copy=MersenneTwister(Int(rng_evaluator.seed[1])+100*(i-1))
        rng_history_copy=MersenneTwister(Int(rng_history.seed[1])+100*(i-1))
        hr.rng = rng_history_copy

        s_initial = initial_state(sim_problem, rng_evaluator_copy)
        s_initial = set_ego_behavior(s_initial, ego_acc)
        o_initial = MLObs(s_initial, problem.dmodel.phys_param.sensor_range, problem.dmodel.phys_param.obs_behaviors)

        ##
        if sim_problem isa POMDP
            if solver isa MLMPCSolver
                updater = make_updater(cor, sim_problem, rng_seed)
                planner = deepcopy(policy)
                srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
                planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
                hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
            else
                updater = LimitedRangeUpdater()
                planner = deepcopy(policy)
                srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
                planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
                hist = simulate(hr, sim_problem, planner, updater, o_initial, s_initial)
            end
        else
            planner = deepcopy(policy)
            planner.solver.n_iterations = 1
            srand(planner, Int(policy.rng.seed[1])+100*(i-1))   #Sets rng seed of planner
            planner.training_phase = false   #Remove random action exploration, always choose the node that was most visited after the MCTS
            hist = simulate(hr, sim_problem, planner, s_initial)
        end

        @show sum(hist.reward_hist)
        @show hist.state_hist[end].x

        action_stats = zeros(1,5)
        for action in hist.action_hist
            if action == MLAction(0.0, 0.0, 1.0)
                action_stats[1] += 1
            elseif action == MLAction(-1.0, 0.0, 1.0)
                action_stats[2] += 1
            elseif action == MLAction(1.0, 0.0, 1.0)
                action_stats[3] += 1
            elseif action == MLAction(0.0, -1.0, 1.0)
                action_stats[4] += 1
            elseif action == MLAction(0.0, 1.0, 1.0)
                action_stats[5] += 1
            end
        end


        open(logs_path*network_to_load*"/prior_policy_result.txt","a") do f
            writedlm(f, [[i, sample_to_load, sum(hist.reward_hist), hist.state_hist[end].x, action_stats[1], action_stats[2], action_stats[3], action_stats[4], action_stats[5] ]], " ")
        end


        # #Produce video
        # frames = Frames(MIME("image/png"), fps=3/pp.dt)
        # @showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
        #     push!(frames, visualize(problem, s, r))
        # end
        # gifname = "./Figs/nn_policy_process_"*string(process)*"_"*network_to_load*"_"*sample_to_load*"__.ogv"
        # write(gifname, frames)

    end

end
