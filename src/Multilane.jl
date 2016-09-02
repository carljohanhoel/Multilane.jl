__precompile__()
module Multilane

import StatsBase: WeightVec, sample

using POMDPs
import POMDPs: actions, discount, isterminal, iterator
import POMDPs: create_action, create_state, rand, reward, create_observation, create_policy
import POMDPs: solve, action
import POMDPs: create_belief, update, initialize_belief
using GenerativeModels
import GenerativeModels: generate_s, generate_sr, initial_state, generate_o, generate_sor

import Distributions: Dirichlet, Exponential, Gamma, rand

import Iterators

import Base: ==, hash, length, vec, +, -, *, .*, ^, .^, .-, /, sqrt, zero, nan, abs, max

# import POMDPToolbox: Particle, ParticleBelief

using DataFrames
using ProgressMeter
using PmapProgressMeter
using POMDPToolbox
import MCTS # so that we can define node_tag, etc.
using RobustMCTS # for RobustMDP
import POMCP # for particle filter

# using Reel
# using AutomotiveDrivingModels
# using AutoViz

import Mustache
import JLD

using Parameters

# package code goes here
export
    PhysicalParam,
    CarState,
    MLState,
    MLAction,
    MLObs,
    BehaviorModel,
    MLMDP,
    MLPOMDP,
    OriginalMDP,
    OriginalRewardModel,
    IDMMOBILModel,
    IDMParam,
    MOBILParam,
    IDMMOBILBehavior,
    MLPhysicalState,
    CarPhysicalState,
    RobustNoCrashMDP,
    FixedBehaviorNoCrashMDP,
    StochasticBehaviorNoCrashMDP,
    RobustMLSolver,
    RobustMLPolicy,
    MLPOMDPSolver,
    MLPOMDPAgent,
    DiscreteBehaviorSet,
    DiscreteBehaviorBelief,
    WeightUpdateParams,
    UniformIDMMOBIL,
    standard_uniform,
    MLMPCSolver,
    MLPOMDPSolver

export
    NoCrashRewardModel,
    NoCrashIDMMOBILModel,
    NoCrashMDP,
    NoCrashPOMDP,
    Simple,
    SimpleSolver,
    BehaviorSolver,
    IDMLaneSeekingSolver

export
    SingleBehaviorSolver,
    SingleBehaviorPolicy,
    single_behavior_state

export
    get_neighborhood, #testing VVV
    get_dv_ds,
    is_lanechange_dangerous,
    get_rear_accel, #testing ^^^
    get_idm_dv,
    get_mobil_lane_change,
    is_crash,
    detect_braking,
    max_braking,
    braking_ids

export #data structure stuff
    ste,
    test_run,
    evaluate,
    merge_results!,
    rerun,
    assign_keys,
    test_run_return_policy,
    lambda,
    TestSet,
    gen_initials,
    run_simulations,
    fill_stats!,
    sbatch_spawn,
    gather_results,
    relaxed_initial_state

export # POMDP belief stuff
    ParticleUpdater,
    create_belief,
    update,
    rand,
    sample,
    initialize_belief,
    BehaviorRootUpdater,
    BehaviorRootUpdaterStub,
    AggressivenessBelief,
    AggressivenessUpdater,
    agg_means,
    agg_stds,
    aggressiveness,
    BehaviorParticleUpdater,
    BehaviorParticleBelief,
    ParticleGenerator,
    param_means,
    param_stds

export
    MaxBrakeMetric,
    NumBehaviorBrakesMetric

export
    include_visualization

include("physical.jl")
include("MDP_types.jl")
include("crash.jl")
include("IDM.jl")
include("MOBIL.jl")
include("behavior.jl")
include("behavior_gen.jl")
include("no_crash_model.jl")
include("metrics.jl")
include("evaluation.jl")
include("test_sets.jl")
include("relax.jl")
include("robust_mdp.jl")
include("pomdp_glue.jl")
include("most_likely_mpc.jl")
include("single_behavior.jl")
include("beliefs.jl")
include("aggressiveness_particle_filter.jl")
include("uniform_particle_filter.jl")
include("heuristics.jl")
include("tree_vis.jl")
include("sherlock.jl")

include_visualization() = include(joinpath(Pkg.dir("Multilane"),"src","visualization.jl"))


end # module
