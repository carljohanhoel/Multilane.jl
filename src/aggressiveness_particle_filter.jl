type AggressivenessBelief <: AbstractDistribution{MLState}
    gen::CorrelatedIDMMOBIL
    physical::MLPhysicalState
    particles::Vector{Vector{Float64}} # First index is the position in physical.env_cars
    weights::Vector{Vector{Float64}}   # Second index is the particle number
end

function rand(rng::AbstractRNG,
                 b::AggressivenessBelief,
                 s::MLState=MLState(b.physical.crashed, Array(CarState, length(b.physical.env_cars))))
    return rand(rng, b, zeros(length(b.physical.env_cars)), s)
end

function rand(rng::AbstractRNG,
                b::AggressivenessBelief,
                sample_noises::Vector{Float64},
                s::MLState=MLState(b.physical.crashed, Array(CarState, length(b.physical.env_cars))))

    s.crashed = b.physical.crashed
    resize!(s.env_cars, length(b.physical.env_cars))
    for i in 1:length(s.env_cars)
        particle = sample(rng, b.particles[i], WeightVec(b.weights[i]))
        nudged = max(min(particle+sample_noises[i]*randn(rng),1.0),0.0)
        s.env_cars[i] = CarState(b.physical.env_cars[i], create_model(b.gen, nudged))
    end
    return s
end

# action(p::Policy, b::AggressivenessBelief) = action(p, b.physical)
action(p::BehaviorPolicy, b::AggressivenessBelief, a::MLAction=MLAction(0.0,0.0)) = action(p, most_likely_state(b))

function most_likely_state(b::AggressivenessBelief)
    s = MLState(b.physical.crashed, Array(CarState, length(b.physical.env_cars)))
    for i in 1:length(s.env_cars)
        ml_ind = indmax(b.weights[i])
        behavior = create_model(b.gen, b.particles[i][ml_ind])
        s.env_cars[i] = CarState(b.physical.env_cars[i], behavior)
    end
    return s
end
agg_means(b::AggressivenessBelief) = [mean(b.particles[i], WeightVec(b.weights[i])) for i in 1:length(b.particles)]
function agg_stds(b::AggressivenessBelief)
    means = agg_means(b)
    stds = Array(Float64, length(b.physical.env_cars))
    for i in 1:length(b.physical.env_cars)
        stds[i] = sqrt(sum(b.weights[i].*(b.particles[i]-means[i]).^2)/sum(b.weights[i]))
    end
    return stds
end


function weights_from_particles!(b::AggressivenessBelief,
                                 problem::NoCrashProblem,
                                 o::MLPhysicalState,
                                 particles,
                                 p::WeightUpdateParams)

    b.physical = o
    resize!(b.weights, length(o.env_cars))
    resize!(b.particles, length(o.env_cars))
    for i in 1:length(o.env_cars)
        # make sure we're not going to be allocating a bunch of memory in the loop
        if isdefined(b.particles, i)
            sizehint!(b.particles[i], length(particles))
            resize!(b.particles[i], 0)
        else
            b.particles[i] = Array(Float64, length(particles))
            resize!(b.particles[i], 0)
        end
        if isdefined(b.weights, i)
            sizehint!(b.weights[i], length(particles))
            resize!(b.weights[i], 0)
        else
            b.weights[i] = Array(Float64, length(particles))
            resize!(b.weights[i], 0)
        end
    end
    for (j, sp) in enumerate(particles)
        isp = 1
        io = 1
        while io <= length(o.env_cars) && isp <= length(sp.env_cars)
            co = o.env_cars[io]
            csp = sp.env_cars[isp]
            if co.id == csp.id
                # sigma_acc = dmodel.vel_sigma/dt
                # dv = acc*dt
                # sigma_v = sigma_dv = dmodel.vel_sigma
                if abs(co.x-csp.x) < 0.2*problem.dmodel.phys_param.lane_length
                    proportional_likelihood = proportional_normal_pdf(csp.vel,
                                                                      co.vel,
                                                                      problem.dmodel.vel_sigma*(1+p.smoothing))
                    if co.y == csp.y
                        push!(b.particles[io], aggressiveness(b.gen, csp.behavior))
                        push!(b.weights[io], proportional_likelihood)
                    elseif abs(co.y - csp.y) < 1.0
                        push!(b.particles[io], aggressiveness(b.gen, csp.behavior))
                        push!(b.weights[io], p.wrong_lane_factor*proportional_likelihood)
                    end # if greater than one lane apart, do nothing
                end
                io += 1
                isp += 1
            elseif co.id < csp.id
                io += 1
            else 
                @assert co.id > csp.id
                isp += 1
            end
        end
    end
   
    return b
end

type AggressivenessUpdater <: Updater{AggressivenessBelief}
    problem::Nullable{NoCrashProblem}
    nb_sims::Int
    resample_noise_factor::Float64 
    params::WeightUpdateParams
    rng::AbstractRNG
end
function set_problem!(u::AggressivenessUpdater, p::Union{POMDP,MDP})
    u.problem = Nullable{NoCrashProblem}(p)
end
function set_rng!(u::AggressivenessUpdater, rng::AbstractRNG)
    u.rng = rng
end

function update(up::AggressivenessUpdater,
                b_old::AggressivenessBelief,
                a::MLAction,
                o::MLPhysicalState,
                b_new::AggressivenessBelief=AggressivenessBelief(get(up.problem).dmodel.behaviors, o,
                                                    Array(Vector{Float64}, length(o.env_cars)),
                                                    Array(Vector{Float64}, length(o.env_cars))))

    particles = Array(MLState, up.nb_sims)
    stds = agg_stds(b_old)
    for i in 1:up.nb_sims
        s = rand(up.rng, b_old, up.resample_noise_factor*stds)
        particles[i] = generate_s(get(up.problem), s, a, up.rng)
    end
    
    weights_from_particles!(b_new, get(up.problem), o, particles, up.params)

    for i in 1:length(o.env_cars)
        if isempty(b_new.weights[i])
            b_new.particles[i] = rand(up.rng, up.nb_sims)
            b_new.weights[i] = ones(up.nb_sims)
        end
    end

    return b_new
end

function initialize_belief(up::AggressivenessUpdater, distribution)
    gen = get(up.problem).dmodel.behaviors
    states = [rand(up.rng, distribution) for i in 1:up.nb_sims]
    particles = Array(Array{Float64}, length(first(states).env_cars))
    weights = Array(Array{Float64}, length(first(states).env_cars))
    for i in 1:length(first(states).env_cars)
        particles[i] = Array(Float64, length(states))
        weights[i] = zeros(length(states))
        for (j,s) in enumerate(states)
            particles[i][j] = aggressiveness(gen, s.env_cars[i].behavior)
        end
    end
    return AggressivenessBelief(gen, MLPhysicalState(first(states)), particles, weights)
end
