#ACC.jl
#Adaptive cruise control model, slightly modified IDM

#############
##ACC Model##
#############

struct ACCParam <: FieldVector{6, Float64}
	a::Float64 #max  comfy acceleration
	b::Float64 #max comfy brake speed
	T::Float64 #desired safety time headway
	v0::Float64 #desired speed
	s0::Float64 #minimum headway (e.g. if x is less than this then you crashed)
	del::Float64 #'accel exponent'
	min_v::Float64 #minimum set speed
	max_v::Float64 #maximum set speed
	min_T::Float64 #minimum set T
	max_T::Float64 #maximum set T
end #ACCParam

StaticArrays.similar_type(::Type{ACCParam}, ::Type{Float64}, ::Size{(6,)}) = ACCParam

nan(::Type{ACCParam}) = ACCParam(NaN, NaN, NaN, NaN, NaN, NaN)


function get_idm_s_star(p::ACCParam, v::Float64, dv::Float64)
    return p.s0 + max(0.,v*p.T+v*dv/(2*sqrt(p.a*p.b)))
end

# dv is positive if vehicles are closing
function get_idm_dv(p::ACCParam,dt::Float64,v::Float64,dv::Float64,s::Float64)
	s_ = get_idm_s_star(p, v, dv)
    @assert p.del == 4.0
    dvdt = p.a*(1.-(v/p.v0)^4 - (s_/s)^2)
	#dvdt = min(max(dvdt,-p.b),p.a)
	return (dvdt*dt)::Float64
end #get_idm_dv



# # these aren't used anymore, but need to be there for the tests
# IDMParam(a::Float64,b::Float64,T::Float64,v0::Float64,s0::Float64;del::Float64=4.) = IDMParam(a,b,T,v0,s0,del)
# function build_cautious_idm(v0::Float64,s0::Float64)
# 	T = 2.
# 	a = 1.
# 	b = 1.
# 	return IDMParam(a,b,T,v0,s0)
# end
#
# function build_aggressive_idm(v0::Float64,s0::Float64)
# 	T = 0.8
# 	a = 2.
# 	b = 2.
# 	return IDMParam(a,b,T,v0,s0)
# end
#
# function build_normal_idm(v0::Float64,s0::Float64)
# 	T = 1.4
# 	a = 1.5
# 	b = 1.5
# 	return IDMParam(a,b,T,v0,s0)
# end
#
# #TODO: use Enum or something to avoid misspelling errors
# function IDMParam(s::AbstractString,v0::Float64,s0::Float64)
# 	if lowercase(s) == "cautious"
# 		return build_cautious_idm(v0,s0)
# 	elseif lowercase(s) == "normal"
# 		return build_normal_idm(v0,s0)
# 	elseif lowercase(s) == "aggressive"
# 		return build_aggressive_idm(v0,s0)
# 	else
# 		error("No such idm phenotype: \"$(s)\"")
# 	end
# end
#
