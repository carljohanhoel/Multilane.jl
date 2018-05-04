# heuristics.jl
# heuristic policies

mutable struct Simple{M} <: Policy #
  mdp::M
  sweeping_up::Bool
end
mutable struct SimpleSolver <: Solver end

Simple(mdp) = Simple(mdp,true)
solve(solver::SimpleSolver, problem::MDP) = Simple(problem)
solve(solver::SimpleSolver, problem::EmbeddedBehaviorMDP) = Simple(problem.base)
solve(solver::SimpleSolver, problem::POMDP) = Simple(problem)
solve(solver::SimpleSolver, problem::AggressivenessBeliefMDP) = Simple(get(problem.up.problem))
solve(solver::SimpleSolver, problem::QMDPWrapper) = Simple(problem.mdp)
solve(solver::SimpleSolver, problem::OutcomeMDP) = Simple(problem.mdp)
POMDPs.updater(::Simple) = POMDPToolbox.FastPreviousObservationUpdater{MLObs}()
create_policy(s::SimpleSolver, problem::MDP) = Simple(problem)
create_policy(s::SimpleSolver, problem::POMDP) = Simple(problem)

set_rng!(solver::SimpleSolver, rng::AbstractRNG) = nothing
Base.srand(p::Simple, s) = p

function action(p::Simple,s::Union{MLState,MLObs})
# lane changes if there is an opportunity
    a = MLAction()
    goal_lane = p.mdp.rmodel.target_lane
    y_desired = goal_lane
    dmodel = p.mdp.dmodel
    lc = sign(y_desired-s.cars[1].y) * dmodel.lane_change_rate
    acc = dmodel.adjustment_acceleration

    #if can't move towards desired lane sweep through accelerating and decelerating

    if is_safe(p.mdp, s, MLAction(0.,lc))
        return MLAction(0.,lc)
    end

    # maintain distance
    nbhd = get_neighborhood(dmodel.phys_param,s,1)

    if nbhd[2] == 0 && nbhd[5] == 0
        return MLAction(0.,0.)
    end

    dist_ahead = nbhd[2] != 0 ? s.cars[nbhd[2]].x - s.cars[1].x : Inf
    dist_behind = nbhd[5] != 0 ? s.cars[nbhd[5]].x - s.cars[1].x : Inf

    sgn = abs(dist_ahead) <= abs(dist_behind) ? -1 : 1

    accel = sgn * acc

    max_accel = max_safe_acc(p.mdp, s, 0.0)

    return MLAction(min(accel, max_accel),0.)
end
action(p::Simple, b::BehaviorBelief) = action(p, b.physical)
function action(pol::Simple, s::QMDPState)
    if s.isstate
        return action(pol, get(s.s))
    else
        return action(pol, get(s.b))
    end
end

mutable struct BehaviorSolver <: Solver
    b::BehaviorModel
    keep_lane::Bool
    rng::AbstractRNG
end
mutable struct BehaviorPolicy <: Policy
    problem::NoCrashProblem
    b::BehaviorModel
    keep_lane::Bool
    rng::AbstractRNG
end
solve(s::BehaviorSolver, p::NoCrashProblem) = BehaviorPolicy(p, s.b, s.keep_lane, s.rng)

function action(p::BehaviorPolicy, s::MLState, a::MLAction=MLAction(0.0,0.0))
    nbhd = get_neighborhood(p.problem.dmodel.phys_param, s, 1)
    acc = gen_accel(p.b, p.problem.dmodel, s, nbhd, 1, p.rng)
    if p.keep_lane
        lc = 0.0
    else
        lc = gen_lane_change(p.b, p.problem.dmodel, s, nbhd, 1)
    end
    return MLAction(acc, lc)
end
action(p::BehaviorPolicy, b::AggressivenessBelief, a::MLAction=MLAction(0.0,0.0)) = action(p, most_likely_state(b))
action(p::BehaviorPolicy, b::BehaviorParticleBelief, a::MLAction=MLAction(0.0,0.0)) = action(p, most_likely_state(b))

mutable struct DeterministicBehaviorSolver <: Solver
    b::BehaviorModel
    keep_lane::Bool
end
mutable struct DeterministicBehaviorPolicy <: Policy
    problem::NoCrashProblem
    b::BehaviorModel
    keep_lane::Bool
end
solve(s::DeterministicBehaviorSolver, p::NoCrashProblem) = DeterministicBehaviorPolicy(p, s.b, s.keep_lane, s.rng)

function action(p::DeterministicBehaviorPolicy, s::MLState, a::MLAction=MLAction(0.0,0.0))
    nbhd = get_neighborhood(p.problem.dmodel.phys_param, s, 1)
    acc = gen_accel(p.b, p.problem.dmodel, s, nbhd, 1) #No rng
    if p.keep_lane
        lc = 0.0
    else
        lc = gen_lane_change(p.b, p.problem.dmodel, s, nbhd, 1)
    end
    return MLAction(acc, lc)
end
action(p::DeterministicBehaviorPolicy, b::AggressivenessBelief, a::MLAction=MLAction(0.0,0.0)) = action(p, most_likely_state(b))
action(p::DeterministicBehaviorPolicy, b::BehaviorParticleBelief, a::MLAction=MLAction(0.0,0.0)) = action(p, most_likely_state(b))

mutable struct IDMLaneSeekingSolver <: Solver
    b::BehaviorModel
    rng::AbstractRNG
end

mutable struct IDMLaneSeekingPolicy <: Policy
    problem::NoCrashProblem
    b::BehaviorModel
    rng::AbstractRNG
end
solve(s::IDMLaneSeekingSolver, p::NoCrashProblem) = IDMLaneSeekingPolicy(p, s.b, s.rng)

function action(p::IDMLaneSeekingPolicy, s::MLState, a::MLAction=MLAction(0.0,0.0))
    nbhd = get_neighborhood(p.problem.dmodel.phys_param, s, 1)
    acc = gen_accel(p.b, p.problem.dmodel, s, nbhd, 1, p.rng)
    # try to positive lanechange
    # lc = problem.dmodel.lane_change_rate * !is_lanechange_dangerous(pp,s,nbhd,1,1)
    lc = p.problem.dmodel.lane_change_rate
    if is_safe(p.problem, s, MLAction(acc, lc))
        return MLAction(acc, lc)
    end
    return MLAction(acc, 0.0)
end



struct OptimisticValue end

function MCTS.estimate_value(v::OptimisticValue, m::NoCrashProblem, s::Union{MLState,MLPhysicalState}, steps::Int)
    rm = m.rmodel
    if rm isa NoCrashRewardModel
        rw = rm.reward_in_target_lane
    elseif rm isa SuccessReward
        rw = 1.0
    else
        error("Unrecognized reward model.")
    end
    dm = m.dmodel
    dt = dm.phys_param.dt
    steps_away = abs(rm.target_lane - first(s.cars).y) / (dm.lane_change_rate * dt)
    if s.x + first(s.cars).vel*steps_away*dt > dm.max_dist
        return 0.0
    elseif dm.lane_terminate
        return discount(m)^steps_away*rw
    else
        gamma = discount(m)
        return gamma^steps_away*rw/(1-gamma)
    end
end

MCTS.estimate_value(v::OptimisticValue, m::MLPOMDP, s::Union{MLState,MLPhysicalState}, h, steps::Int) = estimate_value(v, m, s, steps)
