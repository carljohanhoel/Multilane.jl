using Revise

push!(LOAD_PATH,joinpath("./src"))
using Multilane

using MCTS
using POMDPs
using POMDPModels
using POMDPToolbox

nb_lanes = 4
lane_length = 300.
nb_cars = 20

cor = 0.75

lambda = 0.0
lane_change_cost = 0.0

v_des = 25.0
rmodel = SpeedReward(v_des=v_des, lane_change_cost=lane_change_cost, lambda=lambda)

behaviors = standard_uniform(correlation=cor)   #Sets max/min values of IDM and MOBIL and how they are correlated.
pp = PhysicalParam(nb_lanes, lane_length=lane_length)
dmodel = NoCrashIDMMOBILModel(nb_cars, pp,   #First argument is number of cars
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=30000.0, #1000.0, #ZZZ Remember that the rollout policy must fit within this distance (Not for exit lane scenario)
                              vel_sigma = 0.5,   #0.0   #Standard deviation of speed of inserted cars
                              semantic_actions = true
                             )
mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)   #Third argument is discount factor
rng = MersenneTwister(3)
ego_acc = ACCBehavior(ACCParam(v_des), 1)

initSteps = 100
is = initial_state(mdp, rng, initSteps=initSteps)   #Init random state by simulating 200 steps with standard IDM model
is = set_ego_behavior(is, ego_acc)

cis = convert_state(is, mdp)

##
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)   #Fifth argument semantic action space
is2 = initial_state(pomdp, rng, initSteps=initSteps)   #Init random state by simulating 200 steps with standard IDM model
is2 = set_ego_behavior(is, ego_acc)

cis2 = convert_state(is2, pomdp)
