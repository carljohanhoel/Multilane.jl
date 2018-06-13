function relaxed_initial_state(mdp::NoCrashProblem, steps=200,
                             rng=Base.GLOBAL_RNG;
                             solver=BehaviorSolver(NORMAL, true, rng))

    mdp = deepcopy(mdp)
    mdp.dmodel.fix_number_cars = false   #Allow cars to be added to the simulation
    mdp.dmodel.max_dist = Inf
    mdp.dmodel.brake_terminate_thresh = Inf
    mdp.dmodel.lane_terminate = false
    mdp.dmodel.semantic_actions = false   #Change to direct acceleration action
    pp = mdp.dmodel.phys_param
    is = MLState(0.0, 0.0, CarState[CarState(pp.lane_length/2, rand(rng,1:pp.nb_lanes), pp.v_med, 0.0, NORMAL, 1)])
    sim = HistoryRecorder(max_steps=steps, rng=rng)
    policy = solve(solver, mdp)
    hist = simulate(sim, mdp, policy, is)
    s = last(state_hist(hist))
    s.t = 0.0
    s.x = 0.0
    # @assert s.cars[1].y == 1.0
    @assert isnull(s.terminal)
    return s
end
