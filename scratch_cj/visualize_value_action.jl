# Need to run rerun_trained_model.jl first


#Visualization
#Set time t used for showing tree. Use video to find interesting situations.
t = 0.0
step = convert(Int, t / pp.dt) + 1

state_t = deepcopy(hist.state_hist[step])
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
p_nn = MCTS.init_P(planner.solver.init_P, planner.mdp, state_t, allowed_actions_vec)
v_nn = estimate_value(planner.solved_estimate, planner.mdp, state_t, planner.solver.depth)[1]
p_tree = hist.ainfo_hist[step][:action_distribution]
# @show p_tree
# @show p_nn
# @show v_nn


# print(hist.action_hist[step])
# inchromium(D3Tree(hist.ainfo_hist[step][:tree],init_expand=1))



## Make video

hist_copy = deepcopy(hist)

frames = Frames(MIME("image/png"), fps=3/pp.dt)
step = 0
@showprogress for (s, a, r, sp) in eachstep(hist_copy, "s, a, r, sp")
    s_ego_veh = deepcopy(s.cars[1])
    s_ego_veh_original = deepcopy(s_ego_veh)
    v0 = []
    p0_vec = []
    p0_vec_all_actions = []

    s_vec = []
    for y in 1:0.5:4
        if mod(y,1.0)==0.
            s_ego_veh = CarState(s_ego_veh.x, y, s_ego_veh.vel, 0.0, s_ego_veh.behavior, s_ego_veh.length, s_ego_veh.width, s_ego_veh.id)
            s.cars[1] = s_ego_veh
            push!(s_vec,deepcopy(s))
        else
            for lane_change in [-1,1.]
                s_ego_veh = CarState(s_ego_veh.x, y, s_ego_veh.vel, lane_change, s_ego_veh.behavior, s_ego_veh.length, s_ego_veh.width, s_ego_veh.id)
                s.cars[1] = s_ego_veh
                push!(s_vec,deepcopy(s))
            end
        end
    end

    for state in s_vec
        push!(v0, estimate_value(planner.solved_estimate, planner.mdp, state, planner.solver.depth)[1])

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
    s.cars[1] = s_ego_veh_original

    allowed_actions = actions(planner.mdp, s)   #This is handled a bit weird to make it compatible with existing structure of Multilane.jl
    all_actions = actions(planner.mdp)
    if length(allowed_actions) == length(all_actions)
        allowed_actions_vec = ones(Float64, 1, length(all_actions))
    else
        allowed_actions_vec = zeros(Float64, 1, length(all_actions))
        for idx in collect(allowed_actions.acceptable)
            allowed_actions_vec[idx] = 1.0
        end
    end
    p_nn = MCTS.init_P(planner.solver.init_P, planner.mdp, s, allowed_actions_vec)
    step += 1
    p_tree = hist.ainfo_hist[step][:action_distribution]

    push!(frames, visualize_with_nn(sim_problem,s,a,r,v0,p0_vec, p0_vec_all_actions, p_nn, p_tree))
end
gifname = logs_path*network_to_load*"/Reruns/"*"sample_"*sample_to_load*"_process_"*string(process)*"_options_"*network_to_load*".ogv"
gifname = gifname[1:end-4]*"_options.ogv"
write(gifname, frames)   #The bad quality probably has something to do with the creation of the video. The frames themselves look good. Also, with a smaller fps, it looks good.
