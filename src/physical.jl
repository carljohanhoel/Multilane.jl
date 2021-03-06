#######################
##Physical Parameters##
#######################

mutable struct PhysicalParam
	dt::Float64
	w_car::Float64
	l_car::Float64
    w_truck::Float64
	l_truck::Float64
	v_nominal::Float64
	w_lane::Float64
	v_fast::Float64
	v_slow::Float64
	v_med::Float64
	v_max::Float64
	v_min::Float64
    brake_limit::Float64 # always positive
    nb_lanes::Int
	lane_length::Float64
    sensor_range::Float64 #Only vehicles within this range will be considered by policy
    obs_behaviors::Bool #True if sensors directly measures the behavior of other vehicles, i.e. IDM/MOBIL parameters are not estimated
end

function PhysicalParam(nb_lanes::Int;dt::Float64=0.75,
						w_car::Float64=1.8, #2.0,#1.8,
						l_car::Float64=4.8, #4.0,#4.8,
                        w_truck::Float64=2.5, #2.0,#1.8,#2.5,
						l_truck::Float64=12.0, #4.0,#4.8,#16.5,
						v_nominal::Float64=25.,
						w_lane::Float64=4.,
						v_fast::Float64=30.56,
						v_slow::Float64=19.44,
						v_med::Float64=25.,
						lane_length::Float64=12.,
						v_max::Float64=v_fast+0.,
						v_min::Float64=v_slow-0.,
                        brake_limit::Float64=8., # coefficient of friction of about 0.8
                        sensor_range::Float64=Inf,
                        obs_behaviors::Bool=true
                        )

	assert(v_fast >= v_med)
	assert(v_med >= v_slow)
	assert(v_fast > v_slow)
    return PhysicalParam(dt, w_car, l_car, w_truck, l_truck, v_nominal, w_lane, v_fast, v_slow, v_med, v_max, v_min, brake_limit, nb_lanes, lane_length, sensor_range, obs_behaviors)
end

"""
Returns true if cars at y1 and y2 occupy the same lane
"""
function occupation_overlap(y1::Float64, y2::Float64)
    return abs(y1-y2) < 1.0 || ceil(y1) == floor(y2) || floor(y1) == ceil(y2)
end

"""
Return a Pair{Int} of lanes that the car will occupy at some point in the time step (both lanes could be the same)

Recall that a car can occupy at most two lanes on a single time step
"""
function occupation_lanes(y::Float64, lc::Float64)
    if !isinteger(y)
        return Pair{Int,Int}(floor(Int, y), ceil(Int, y))
    else # y is an integer
        return Pair{Int,Int}(y, y + sign(lc))
    end
end

"""
Return true if cars will occupy the same lane at some time in the time step
"""
function occupation_overlap(y1::Float64, lc1::Float64, y2::Float64, lc2::Float64)
    lanes1 = occupation_lanes(y1, lc1)
    lanes2 = occupation_lanes(y2, lc2)
    return lanes1[1] in lanes2 || lanes1[2] in lanes2
end
