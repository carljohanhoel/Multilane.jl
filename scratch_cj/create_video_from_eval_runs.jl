push!(LOAD_PATH,joinpath("./src"))

using JLD

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
using D3Trees
##

# @show scenario = "continuous_driving"
@show scenario = "exit_lane"

#Code for loading saved evaluation history, visualizing MCTS tree and producing video
logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"
# log_dir = logs_path*"181119_180615_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Added_set_V_set_T_ego_state"
# log_dir = logs_path*"181126_154437_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim"
# log_dir = logs_path*"181126_155336_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim_Weights_1_10"
# log_dir = logs_path*"181215_123610_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim_Bigger_net"

# log_dir = logs_path*"181130_160730_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim"
# log_dir = logs_path*"181203_174746_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim_Terminal_state_Est_v0"
log_dir = logs_path*"181215_121952_driving_exit_lane_Cpuct_0p5_Dpw_0p3_Big_replay_Truck_dim_Terminal_state_Est_v0_R_plus_19"


eval_files = []
all_files = readdir(log_dir)
for file_name in all_files
    if startswith(file_name, "eval_hist")
        if !in("video"*file_name[10:end-4]*".ogv",all_files)
            push!(eval_files,file_name)
        end
    end
end

if length(eval_files) == 0
    print("No (new) files to process\n")
end



#Below code is just needed to create the problem, whose physical parameters are needed for the visualization
simple_run = false
n_workers = 1
include("parameters.jl")
# include(logs_path*network_to_load*"/code/scratch_cj/parameters.jl")

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

##
# #Visualization
# #Set time t used for showing tree. Use video to find interesting situations.
# t = 3.0
# step = convert(Int, t / pp.dt) + 1
# write_to_png(visualize(problem,hist_loaded.state_hist[step],hist_loaded.reward_hist[step]),"./Figs/state_at_t.png")
# print(hist_loaded.action_hist[step])
# inchromium(D3Tree(hist_loaded.ainfo_hist[step][:tree],init_expand=1))


for eval_file in eval_files
    hist_loaded = JLD.load(log_dir*"/"*eval_file)
    hist_loaded = hist_loaded["hist"]

    #Produce video
    frames = Frames(MIME("image/png"), fps=10/pp.dt)
    @showprogress for (s, ai, r, sp) in eachstep(hist_loaded, "s, ai, r, sp")
        push!(frames, visualize(problem, s, r))
    end
    gifname = log_dir*"/video"*eval_file[10:end-4]*".ogv"
    write(gifname, frames)
end
