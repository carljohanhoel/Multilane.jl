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
    v_des::Float64                  = 33.3 # always positive
    lane_change_cost::Float64       = 0.0 # always positive
    brake_penalty_thresh::Float64   = 4.0  # always positive (also used when calculating actions space of node)
    target_lane::Int                = typemax(Int) #Not used here. But needed for condition that terminates simulation when target lane is reached (if using a target lane).
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

function reward(mdp::MLMDP{MLState, MLAction, D, SpeedReward},
          s::MLState,
          ::MLAction,
          sp::MLState) where D<:AbstractMLDynamicsModel

    v_ego = sp.cars[1].vel
    # r = 1/(1 + ((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)^2)
    r = 1 - abs((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)

    if !isinteger(sp.cars[1].y)
        r -= mdp.rmodel.lane_change_cost
    end
    # r += sp.cars[1].y == 4 ? 1 : 0 #Just for testing

    return r
end

function reward(mdp::MLPOMDP{MLState, MLAction, MLPhysicalState, D, SpeedReward},
          s::MLState,
          ::MLAction,
          sp::MLState) where D<:AbstractMLDynamicsModel

    v_ego = sp.cars[1].vel
    # r = 1/(1 + ((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)^2)
    r = 1 - abs((v_ego-mdp.rmodel.v_des)/mdp.rmodel.v_des)

    if !isinteger(sp.cars[1].y)
        r -= mdp.rmodel.lane_change_cost
    end
    # r += sp.cars[1].y == 4 ? 1 : 0 #Just for testing

    return r
end
