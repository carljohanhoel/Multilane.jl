# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

# parallel_version = true   #Test code in parallel mode
parallel_version = false

# simple_run = true
simple_run = false

tree_in_info = true

sample_to_load = "26031"
network_to_load = "181024_154644_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03"
# network_to_load = "181024_155209_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample"
logs_path = "/home/cj/2018/Stanford/Code/Multilane.jl/Logs/"


include("simulation_setup.jl")


##

#Load state from log
# saved_data = JLD.load("./Logs/states_run_i.jld")
# saved_data = JLD.load("./Logs/saved_states_no_vehicles.jld")
# rec_states = saved_data["state_hist"]
saved_data = JLD.load(logs_path*network_to_load*"/eval_hist_process_5_step_"*sample_to_load*".jld")
rec_states = saved_data["hist"].state_hist
t = 0.0
step = convert(Int, t / pp.dt) + 1
test_state = rec_states[step]

#Change acc time state
acc = test_state.cars[1].behavior.p_idm
acc2 = [value for (idx,value) in enumerate(acc)]
# acc2[4] = 24.0
# acc2[3] =   0.5
# test_state.cars[1] = CarState(test_state.cars[1].x,test_state.cars[1].y,test_state.cars[1].vel,test_state.cars[1].lane_change, ACCBehavior(ACCParam(acc2),1), test_state.cars[1].length, test_state.cars[1].width, test_state.cars[1].id)
# test_state.cars[1] = CarState(test_state.cars[1].x,1.0,test_state.cars[1].vel,test_state.cars[1].lane_change, ACCBehavior(ACCParam(acc2),1), test_state.cars[1].length, test_state.cars[1].width, test_state.cars[1].id)
# test_state.cars[1] = CarState(test_state.cars[1].x,3.5,test_state.cars[1].vel,1.0, ACCBehavior(ACCParam(acc2),1), test_state.cars[1].length, test_state.cars[1].width, test_state.cars[1].id)
# test_state.cars[1] = CarState(test_state.cars[1].x,3.0,25.36,test_state.cars[1].lane_change, ACCBehavior(ACCParam(acc2),1), test_state.cars[1].length, test_state.cars[1].width, test_state.cars[1].id)

#Run MCTS
policy.training_phase = false #Remove Dirichlet noise on prior probabilities
a, ai = action_info(policy, test_state)

write_to_png(visualize(sim_problem,rec_states[step],0.0),"./Figs/state_at_t.png")
inchromium(D3Tree(ai[:tree],init_expand=1))

##Plot value for different lanes
s_ego_veh = test_state.cars[1]
v0 = zeros(7,1)
p0_vec = zeros(7,5)
for i in 1:7
    y = i*0.5 + 0.5
    s_ego_veh = CarState(s_ego_veh.x, y, s_ego_veh.vel, (mod(y,1.0)>0)*1.0, s_ego_veh.behavior, s_ego_veh.length, s_ego_veh.width, s_ego_veh.id)
    test_state.cars[1] = s_ego_veh
    v0[i] = estimate_value(policy.solved_estimate, policy.mdp, test_state, policy.solver.depth)[1]

    allowed_actions = actions(policy.mdp, test_state)   #This is handled a bit weird to make it compatible with existing structure of Multilane.jl
    all_actions = actions(policy.mdp)
    if length(allowed_actions) == length(all_actions)
        allowed_actions_vec = ones(Float64, 1, length(all_actions))
    else
        allowed_actions_vec = zeros(Float64, 1, length(all_actions))
        for idx in collect(allowed_actions.acceptable)
            allowed_actions_vec[idx] = 1.0
        end
    end
    p0_vec[i,:] = MCTS.init_P(policy.solver.init_P, policy.mdp, test_state, allowed_actions_vec)
end
write_to_png(visualize_with_nn(sim_problem,test_state,0.0,v0,p0_vec),"./Figs/estimated_values.png")
