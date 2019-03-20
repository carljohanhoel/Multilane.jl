# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

# @show scenario = "continuous_driving"
@show scenario = "exit_lane"

# parallel_version = true   #Test code in parallel mode
parallel_version = false

# simple_run = true
simple_run = false

tree_in_info = false
# tree_in_info = true

logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"

network_to_load = "190128_142353_driving_exit_lane_Cpuct_0p1_Bigger_net_New_action_space_No_batchnorm"

sample_to_load = "15038" #This refers to network sample

process = "3"
eval_log = "15038" #This should correspond to sample_to_load
eval_file = "eval_hist_process_"*process*"_step_"*eval_log*".jld"


include("simulation_setup.jl")

planner = deepcopy(solve(solver, sim_problem))

# for eval_file in eval_files

log_dir = logs_path*network_to_load
hist_loaded = JLD.load(log_dir*"/"*eval_file)
hist_loaded = hist_loaded["hist"]

hist_copy = deepcopy(hist_loaded)

if true
    for state in hist_copy.state_hist
        number_of_cars = length(state.cars)
        for i in 1:number_of_cars-1
            pop!(state.cars)
        end
    end
end

value_matrix = []
action_matrix = []
x_vector = Float64[]
frames = Frames(MIME("image/png"), fps=3/pp.dt)
step = 0

s_ego_veh = deepcopy(hist_copy.state_hist[1].cars[1])
s = deepcopy(hist_copy.state_hist[1])
ego_behavior = ACCBehavior(ACCParam(1.4,2.0,0.5,25.0,0.0,4.0,15.0,25.0,2.0,0.5,2.5,1.0,2.5),1)
# s_other_vehicle1 = CarState(900.,1.,20.,0.,IDMMOBILBehavior(IDMParam(0.8,1.,2.,20.,4.,4.),MOBILParam(0.,1.,0.2),2),4.8,1.8,2)
# s_other_vehicle2 = CarState(900.,2.,20.,0.,IDMMOBILBehavior(IDMParam(0.8,1.,2.,20.,4.,4.),MOBILParam(0.,1.,0.2),3),4.8,1.8,3)
# push!(s.cars,s_other_vehicle1)
# push!(s.cars,s_other_vehicle2)
a = deepcopy(hist_copy.action_hist[1])
r = 1.0

# ###############################3
# policy.solver.n_iterations = 10
# ###########################

for x in 0:50.:1000
    v0 = []
    a_vec = []
    p0_vec = []
    p0_vec_all_actions = []
    s_vec = []
    for y in 1:1:4
        if mod(y,1.0)==0.
            s_ego_veh = CarState(s.cars[1].x, y, 25.0, 0.0, ego_behavior, s_ego_veh.length, s_ego_veh.width, s_ego_veh.id)
            s.cars[1] = s_ego_veh
            s.x = x
            push!(s_vec,deepcopy(s))
        else
            for lane_change in [-1,1.]
                s_ego_veh = CarState(s.cars[1].x, y, 25.0, lane_change, ego_behavior, s_ego_veh.length, s_ego_veh.width, s_ego_veh.id)
                s.cars[1] = s_ego_veh
                s.x = x
                push!(s_vec,deepcopy(s))
            end
        end
    end

    for state in s_vec
        push!(v0, estimate_value(planner.solved_estimate, planner.mdp, state, planner.solver.depth)[1])

        policy.training_phase = false #Remove Dirichlet noise on prior probabilities
        a, ai = action_info(policy, state)
        push!(a_vec,a)


        allowed_actions = actions(planner.mdp, state)   #This is handled a bit weird to make it compatible with existing structure of Multilane.jl
        all_actions = actions(planner.mdp)
        if length(allowed_actions) == length(all_actions)
            allowed_actions_vec = ones(Float64, 1, length(all_actions))
        else
            allowed_actions_vec = zeros(Float64, 1, length(all_actions))
            for idx in collect(allowed_actions.acceptable)
                allowed_actions_vec[idx] = 1.0
            end
        end
        push!(p0_vec, MCTS.init_P(planner.solver.init_P, planner.mdp, state, allowed_actions_vec))
        #Raw estimation, without removing forbidden actions
        converted_state = MCTS.convert_state(state, planner.mdp)
        dist, value = estimator.py_class[:forward_pass](converted_state)
        push!(p0_vec_all_actions, dist)
    end

    # #This is just for video...
    # allowed_actions = actions(planner.mdp, s)   #This is handled a bit weird to make it compatible with existing structure of Multilane.jl
    # all_actions = actions(planner.mdp)
    # if length(allowed_actions) == length(all_actions)
    #     allowed_actions_vec = ones(Float64, 1, length(all_actions))
    # else
    #     allowed_actions_vec = zeros(Float64, 1, length(all_actions))
    #     for idx in collect(allowed_actions.acceptable)
    #         allowed_actions_vec[idx] = 1.0
    #     end
    # end
    # p_nn = MCTS.init_P(planner.solver.init_P, planner.mdp, s, allowed_actions_vec)
    # step += 1
    # p_tree = p_nn*NaN
    # push!(frames, visualize_with_nn(sim_problem,s,a,r,v0,p0_vec, p0_vec_all_actions, p_nn, p_tree))


    push!(action_matrix,a_vec)
    push!(value_matrix,v0)
    push!(x_vector,s.x)
end

# gifname = log_dir*"/video_nn_value_map_"*eval_file[10:end-4]*".ogv"
# write(gifname, frames)   #The bad quality probably has something to do with the creation of the video. The frames themselves look good. Also, with a smaller fps, it looks good.

vv = []
for item in value_matrix
   push!(vv,convert(Array{Float64,1},item))
end
# open(logs_path*network_to_load*"/Reruns/"*"values_empty_road.txt","a") do f
#     writedlm(f, vv, " ")
# end
# open(logs_path*network_to_load*"/Reruns/"*"x_empty_road.txt","a") do f
#     writedlm(f, x_vector, " ")
# end

a_num = Array{Int}[]
for a_vec in action_matrix
    a_num_vec = Int[]
    for a in a_vec
        if a.acc == 0. && a.lane_change == 0.
            a_ = 1
        elseif a.acc == -1. && a.lane_change == 0.
            a_ = 2
        elseif a.acc == 1. && a.lane_change == 0.
            a_ = 3
        elseif a.acc == 0. && a.lane_change == -1.
            a_ = 4
        elseif a.acc == 0. && a.lane_change == 1.
            a_ = 5
        else
            println("ERROR")
        end
        push!(a_num_vec,a_)
    end
    push!(a_num, a_num_vec)
end

open(logs_path*network_to_load*"/Reruns/"*"a_empty_road.txt","a") do f
    writedlm(f, a_num, " ")
end

open(logs_path*network_to_load*"/Reruns/"*"x_action_map.txt","a") do f
    writedlm(f, x_vector, " ")
end




# end
