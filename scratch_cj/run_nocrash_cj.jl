push!(LOAD_PATH,joinpath("./src"))

using Revise

using Multilane
using MCTS

using POMDPToolbox
using POMDPs
using POMCPOW

#For viz
using AutoViz
using Reel
using ProgressMeter
using AutomotiveDrivingModels
using ImageView
using Images

# @show scenario = "continuous_driving"
@show scenario = "exit_lane"

# simple_run = true
simple_run = false
n_workers = 1

include("parameters.jl")

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


i=3
rng_evaluator=MersenneTwister(rng_seed+2 +100*(i-1))
rng_history=MersenneTwister(rng_seed+4 +100*(i-1))


v_des = 25.0
behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 0.1), 1)
policy = Multilane.DeterministicBehaviorPolicy(mdp, behavior, false)   #Sets up behavior of ego vehicle for the simulation. False referes to that lane changes are allowed.
ego_acc = ACCBehavior(ACCParam(v_des), 1)
# policy = Multilane.DeterministicBehaviorPolicy(mdp, ego_acc, true)   #No lane changes
if scenario == "exit_lane"
    solver = IDMLaneSeekingSolver(behavior)
    # solver = SimpleSolver()
    policy = solve(solver, mdp)
end

s = initial_state(mdp::NoCrashMDP, rng_evaluator, initSteps=initSteps) #Creates inital state by first initializing only ego vehicle and then running simulatio for 200 steps, where additional vehicles are randomly added.
is = set_ego_behavior(s, ego_acc)
write_to_png(visualize(mdp,s,0.0),"Figs/initState.png")

sim = HistoryRecorder(rng=rng_history, max_steps=episode_length, show_progress=true) # initialize a random number generator
hist = simulate(sim, mdp, policy, s)   #Run simulation, here with standard IDM&MOBIL model as policy

println("sim done")


frames = Frames(MIME("image/png"), fps=3/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(mdp, s, r))
end
gifname = "./Figs/testViz.ogv"
write(gifname, frames)
println("video done")
