#Runs simulation with MDP and POMDP settings, but uses no MCTS to control the ego vehicle.

push!(LOAD_PATH,joinpath("./src"))
include("../src/Multilane.jl")

using Revise #To allow recompiling of modules withhout restarting julia

using Multilane
using MCTS
using POMDPs
using POMDPToolbox
using Base.Test

#For viz
using POMCPOW
using AutoViz
using Reel
using ProgressMeter
using AutomotiveDrivingModels
using ImageView
using Images
# include("../src/visualization.jl")

##
#Set up problem configuration
nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=400.) #2.=>col_length=8   #Sets parameters of lanes and cars. Quite a few standard parameters are set here.
_discount = 1.
nb_cars=20

rmodel = NoCrashRewardModel()   #Sets parameters of reward model
behaviors = standard_uniform(correlation=0.75)   #Sets max/min values of IDM and MOBIL and how they are correlated.
dmodel = NoCrashIDMMOBILModel(nb_cars, pp, behaviors=behaviors)   #Sets up simulation model parameters.
dmodel.max_dist = 100000   #Make svisualization fail if max_dist is set to the default Inf
mdp = NoCrashMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, _discount, true);   #Sets the mdp, which inherits from POMDPs.jl
rng = MersenneTwister(6)

initSteps = 150
s = initial_state(mdp::NoCrashMDP, rng, initSteps=initSteps) #Creates inital state by first initializing only ego vehicle and then running simulatio for 200 steps, where additional vehicles are randomly added.
# @show s.cars[1]
#visualize(mdp,s,MLAction(0,0))
write_to_png(visualize(mdp,s,0.0),"Figs/initState.png")

v_des = 25.0
behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 0.1), 1)
policy = Multilane.DeterministicBehaviorPolicy(mdp, behavior, false)   #Sets up behavior of ego vehicle for the simulation. False referes to that lane changes are allowed.

sim = HistoryRecorder(rng=rng, max_steps=500, show_progress=true) # initialize a random number generator
hist = simulate(sim, mdp, policy, s)   #Run simulation, here with standard IDM&MOBIL model as policy

frames = Frames(MIME("image/png"), fps=10/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(mdp, s, r))
end
gifname = "./Figs/testViz.ogv"
write(gifname, frames)

# check for crashes
for i in 1:length(state_hist(hist))-1
    if is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
        println("Crash:")
        println("mdp = $mdp\n")
        println("s = $(state_hist(hist)[i])\n")
        println("a = $(action_hist(hist)[i])\n")
        println("Saving gif...")
        f = write_tmp_gif(mdp, hist)
        println("gif written to $f")
    end
    @test !is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
end

# for i in 1:length(sim.state_hist)-1
#     if is_crash(mdp, sim.state_hist[i], sim.state_hist[i+1])
#         visualize(mdp, sim.state_hist[i], sim.action_hist[i], sim.state_hist[i+1], two_frame_crash=true)
#         # println(repr(mdp))
#         # println(repr(sim.state_hist[i]))
#         println("Crash after step $i")
#         println("Chosen Action: $(sim.action_hist[i])")
#         println("Available actions:")
#         for a in actions(mdp, sim.state_hist[i], actions(mdp))
#             println(a)
#         end
#         println("Press Enter to continue.")
#         readline(STDIN)
#     end
# end

##
#---------------------------------------


behaviors = standard_uniform(correlation=0.75)   #Creates behaviors
dmodel = NoCrashIDMMOBILModel(dmodel, behaviors)   #Changes the behaviors in dmodel to the specified ones
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, _discount, true);

rng = MersenneTwister(6)

initSteps = 150
s = initial_state(mdp::NoCrashMDP, rng, initSteps=initSteps)

v_des = 25.0
behavior = IDMMOBILBehavior(IDMParam(1.4, 2.0, 1.5, v_des, 2.0, 4.0), MOBILParam(0.5, 2.0, 0.1), 1)
policy = Multilane.DeterministicBehaviorPolicy(mdp, behavior, false)   #Sets up behavior of ego vehicle for the simulation. False referes to that lane changes are allowed.

sim = HistoryRecorder(rng=rng, max_steps=500) # initialize a random number generator

wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)
up = BehaviorParticleUpdater(pomdp, 1000, 0.1, 0.1, wup, MersenneTwister(50000))

@time hist = simulate(sim, pomdp, policy, up, MLPhysicalState(s), s)
@show n_steps(hist)

# check for crashes
for i in 1:length(state_hist(hist))-1
    if is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
        println("Crash:")
        println("mdp = $mdp\n")
        println("s = $(state_hist(hist)[i])\n")
        println("a = $(action_hist(hist)[i])\n")
        println("Saving gif...")
        f = write_tmp_gif(mdp, hist)
        println("gif written to $f")
    end
    @test !is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
end


frames = Frames(MIME("image/png"), fps=10/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, ai, r, sp")
    push!(frames, visualize(pomdp, s, r))
end
gifname = "./Figs/testViz2.ogv"
write(gifname, frames)
