struct LimitedRangeSolver <: Solver
    solver
end

function solve(sol::LimitedRangeSolver, p::NoCrashProblem)
    mdp = NoCrashMDP{typeof(p.rmodel), typeof(p.dmodel.behaviors)}(p.dmodel, p.rmodel, p.discount, p.throw) # make sure an MDP
    return LimitedRangePolicy(solve(sol.solver, mdp))
end

struct LimitedRangePolicy{P<:Policy} <: Policy
    planner::P
end

Base.srand(p::LimitedRangePolicy, s) = srand(p.planner, s)
action_info(p::LimitedRangePolicy, b) = action_info(p.planner, b)
action(p::LimitedRangePolicy, b) = first(action_info(p, b))

##

mutable struct LimitedRangeUpdater <: Updater
end
# function set_problem!(u::BehaviorParticleUpdater, p::Union{POMDP,MDP})
#     u.problem = Nullable{NoCrashProblem}(p)
# end
# function set_rng!(u::BehaviorParticleUpdater, rng::AbstractRNG)
#     u.rng = rng
# end

##


function update(up::LimitedRangeUpdater,
                b_old::MLState,
                a::MLAction,
                o::MLState)
    return o

end


function initialize_belief(up::LimitedRangeUpdater, state::MLState)
    return state
end
