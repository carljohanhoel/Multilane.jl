# Need to run rerun_trained:model.jl first


#Visualization
#Set time t used for showing tree. Use video to find interesting situations.
t = 27.
step = convert(Int, t / pp.dt) + 1
state_t = deepcopy(hist.state_hist[step])


state_t_saved = deepcopy(state_t)
s_ego_veh = state_t.cars[1]
v0 = zeros(7,1)
p0_vec = zeros(7,5)
for i in 1:7
    y = i*0.5 + 0.5
    s_ego_veh = CarState(s_ego_veh.x, y, s_ego_veh.vel, (mod(y,1.0)>0)*1.0, s_ego_veh.behavior, s_ego_veh.id)
    state_t.cars[1] = s_ego_veh
    v0[i] = estimate_value(planner.solved_estimate, planner.mdp, state_t, planner.solver.depth)[1]

    allowed_actions = actions(planner.mdp, state_t)   #This is handled a bit weird to make it compatible with existing structure of Multilane.jl
    all_actions = actions(planner.mdp)
    if length(allowed_actions) == length(all_actions)
        allowed_actions_vec = ones(Float64, 1, length(all_actions))
    else
        allowed_actions_vec = zeros(Float64, 1, length(all_actions))
        for idx in collect(allowed_actions.acceptable)
            allowed_actions_vec[idx] = 1.0
        end
    end
    p0_vec[i,:] = MCTS.init_P(planner.solver.init_P, planner.mdp, state_t, allowed_actions_vec)
end


# p0_vec = MCTS.init_P(planner.solver.init_P, planner.mdp, state_t, allowed_actions_vec)
# v0 = estimate_value(planner.solved_estimate, planner.mdp, state_t, planner.solver.depth)[1]



write_to_png(visualize_with_nn(sim_problem,state_t,hist.reward_hist[step],v0,p0_vec),"./Figs/options.png")
print(hist.action_hist[step])
inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))
# inchromium(D3Tree(hist.ainfo_hist[step][:tree],hist.state_hist[step],init_expand=1))   #For MCTS (not DPW)
