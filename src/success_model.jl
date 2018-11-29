"""
Gives a reward of +1 if sp is the target lane
"""
struct TargetLaneReward <: AbstractMLRewardModel
    target_lane::Int
end
function reward(mdp::MLMDP{MLState, MLAction, D, TargetLaneReward},
          s::MLState,
          ::MLAction,
          sp::MLState) where D<:AbstractMLDynamicsModel
    return isnull(s.terminal) && sp.cars[1].y == mdp.rmodel.target_lane
end

function reward(mdp::MLPOMDP{MLState, MLAction, MLPhysicalState, D, TargetLaneReward},
          s::MLState,
          ::MLAction,
          sp::MLState) where D<:AbstractMLDynamicsModel
    return isnull(s.terminal) && sp.cars[1].y == mdp.rmodel.target_lane
end

"""
Reward of +1 on transition INTO target lane, -lambda on unsafe transitions, -lane_change_cost if sp between lanes
"""
@with_kw struct SuccessReward <: AbstractMLRewardModel
    lambda::Float64                 = 1.0  # always positive
    target_lane::Int                = 4
    brake_penalty_thresh::Float64   = 4.0  # always positive
    speed_thresh::Float64           = 15.0 # always positive
    lane_change_cost::Float64       = 0.0 # always positive
end

function reward(p::NoCrashProblem{SuccessReward}, s::MLState, ::MLAction, sp::MLState)
    if sp.cars[1].y == p.rmodel.target_lane && s.cars[1].y != p.rmodel.target_lane
        r = 1.0
    else
        r = 0.0
    end
    min_speed = minimum(c.vel for c in sp.cars)
    nb_brakes = detect_braking(p, s, sp)
    if nb_brakes > 0 || min_speed < p.rmodel.speed_thresh
        r -= p.rmodel.lambda
    end
    if !isinteger(sp.cars[1].y)
        r -= p.rmodel.lane_change_cost
    end
    return r
end


"""
Simple speed reward
"""
@with_kw struct SpeedReward <: AbstractMLRewardModel
    v_des::Float64                  = 25.0 # always positive
    lane_change_cost::Float64       = 0.0 # always positive
    brake_penalty_thresh::Float64   = 4.0  # always positive (also used when calculating actions space of node)
    target_lane::Int                = typemax(Int) #Not used here. But needed for condition that terminates simulation when target lane is reached (if using a target lane).
    lambda::Float64                 = 0.0  # always positive
end

# function reward(p::NoCrashProblem{SpeedReward}, s::MLState, ::MLAction, sp::MLState)
#
#     v_ego = sp.cars[1].vel
#     r = 1/(1 + ((v_ego-p.rmodel.v_des)/p.rmodel.v_des)^2)
#
#     if !isinteger(sp.cars[1].y)
#         r -= p.rmodel.lane_change_cost
#     end
#
#     return r
# end

function reward(mdp::Union{ MLMDP{MLState, MLAction, D, SpeedReward}, MLPOMDP{MLState, MLAction, MLPhysicalState, D, SpeedReward}, MLPOMDP{MLState, MLAction, MLState, D, SpeedReward} },
          s::MLState,
          a::MLAction,
          sp::MLState) where D<:AbstractMLDynamicsModel

    v_ego = sp.cars[1].vel
    v_set = sp.cars[1].behavior.p_idm.v0
    # r = 1/(1 + ((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)^2)
    speed_penalty = abs((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)
    set_speed_penalty = abs((v_set-mdp.rmodel.v_des)/mdp.rmodel.v_des)
    r = 1 - speed_penalty - set_speed_penalty

    # if !isinteger(sp.cars[1].y) #This punishes being between lanes. But no negative reward is given for the final step, stepping into a new lane (or returning to the original lane)
    #     r -= mdp.rmodel.lane_change_cost
    # end

    if abs(a.lane_change) > 0.0 && isinteger(s.cars[1].y)   #Only cost for lane change when initiating one. Then no more for finishing or aborting. (But could this cause problems with hesitation, changing back and forth? Maybe have cost for initiating and for changing direction? On the other hand, since there is a cost for initiating, there neeed to be a clear incentive to start a lane change, and to reach the goal. So should probably be fine as it is.)
        r -= mdp.rmodel.lane_change_cost
    end
    if !isinteger(s.cars[1].y) && a.lane_change != s.cars[1].lane_change #Cost for regretting a lane change
        r -= mdp.rmodel.lane_change_cost
    end

    nb_brakes = detect_braking(mdp, s, sp)
    if nb_brakes > 0
        r -= mdp.rmodel.lambda
    end

    if mdp.rmodel.target_lane < 5 #exit lane scenario
        if !isnull(sp.terminal)
            if sp.cars[1].y == mdp.rmodel.target_lane
                r = 20
            else
                r = 0
            end
        end
    end

    # r += sp.cars[1].y == 4 ? 1 : 0 #Just for testing

    return r
end

function max_min_cum_reward(mdp::Union{ MLMDP{MLState, MLAction, D, SpeedReward}, MLPOMDP{MLState, MLAction, MLPhysicalState, D, SpeedReward}, MLPOMDP{MLState, MLAction, MLState, D, SpeedReward} }) where D<:AbstractMLDynamicsModel
    #Not properly defined, but fine for now
    # v_min = 0
    if mdp.rmodel.target_lane == 1     #Exit lane case
        v_min = 0.
        v_max = 1/(1-mdp.discount)
    else                                #Continuous driving case
        v_min = 0.5/(1-mdp.discount)
        v_max = 1/(1-mdp.discount)
    end
    return [v_min, v_max]
end


# function reward(mdp::MLPOMDP{MLState, MLAction, MLPhysicalState, D, SpeedReward},
#           s::MLState,
#           ::MLAction,
#           sp::MLState) where D<:AbstractMLDynamicsModel
#
#     v_ego = sp.cars[1].vel
#     # r = 1/(1 + ((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)^2)
#     r = 1 - abs((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)
#
#     if !isinteger(sp.cars[1].y)
#         r -= mdp.rmodel.lane_change_cost
#     end
#     # r += sp.cars[1].y == 4 ? 1 : 0 #Just for testing
#
#     return r
# end


function MCTS.create_eval_log(p::Union{NoCrashMDP,NoCrashPOMDP,NoCrashPOMDP_lr},hist::Union{POMDPToolbox.MDPHistory,POMDPToolbox.POMDPHistory}, process_id::Int, step::Int)
    log = Array{Float64}[]
    push!(log,[process_id, step, sum(hist.reward_hist)])
    push!(log,hist.reward_hist)
    push!(log, [hist.state_hist[i].x for i=1:length(hist.state_hist)]) #Ego x position
    push!(log, [hist.state_hist[i].cars[1].y for i=1:length(hist.state_hist)]) #Ego y position
    push!(log, [hist.state_hist[i].cars[1].vel for i=1:length(hist.state_hist)]) #Ego velocity
    push!(log, [hist.state_hist[i].cars[1].lane_change for i=1:length(hist.state_hist)]) #Ego lane change
    push!(log, [hist.state_hist[i].cars[1].behavior.p_idm[3] for i=1:length(hist.state_hist)]) #Ego set time gap
    push!(log, [hist.state_hist[i].cars[1].behavior.p_idm[4] for i=1:length(hist.state_hist)]) #Ego set speed
    actions = Array{Float64}(length(hist.action_hist))
    if hist.action_hist[1].semantic == 1
        for (i,action) in enumerate(hist.action_hist)
            as = NoCrashSemanticActionSpace(p).actions
            actions[i] = find(as .== action)[1]
        end
    end
    push!(log, actions) #Action number


    write_to_png(visualize(p,hist.state_hist[1],0),"./tmpFigs/state_at_t0_i"*string(process_id-2)*".png")

    return log #Transpose to row vector
end
