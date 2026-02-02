using DiffEqCallbacks
using JLD2
using ModelingToolkit
using OrdinaryDiffEq
using SteadyStateDiffEq

using ModelingToolkit: t_nounits as t, D_nounits as D

# Define the helper functions
θp(L, K) = L^2/(K + L^2)
θm(L, K) = 1/(1 + (L/K)^2)

function generate_callbacks_learning(sys, tspan, delay, duration)
	# Create event time points
	t_start_events = (tspan[1]:delay:tspan[2])[2:end-1]
	t_end_events = t_start_events .+ duration

	# Find the index of x in the state variables
	x_pos = findfirst(isequal(sys.x), unknowns(sys))

	# Find the index of G in the state variables
	G_pos = findfirst(isequal(sys.G), unknowns(sys))

	# Find extrema.
	dG_zero = (u, t, integrator) -> let
		du = similar(u)
		integrator.f(du, u, integrator.p, t)
		du[G_pos]
	end
	
	# Create callbacks for events
	cbs = CallbackSet(
		PositiveDomain(save = false),
		PresetTimeCallback(
			t_start_events,
			(integrator) -> integrator.u[x_pos] = 1,
		),
		PresetTimeCallback(
			t_end_events,
			(integrator) -> integrator.u[x_pos] = 0,
		),
		# Find maxima.
		ContinuousCallback(dG_zero, nothing, i -> nothing),
	)
	events = collect(zip(t_start_events, t_end_events))

	return (cbs, events)
end

function solve_problem_learning(def_prob, def_ssprob, α, γ)
	# Solve system for steady state with no input.
	ssprob = remake(def_ssprob; p = [:α => α, :γ => γ])
	sssol = solve(ssprob, DynamicSS(Rodas5P());
		abstol = 1e-8,
		reltol = 1e-8,
	)

	# Solve system from steady state with callbacks.
	prob = remake(def_prob; u0 = sssol.u, p = [:α => α, :γ => γ])
	sol = solve(prob, AutoTsit5(Rosenbrock23());
		abstol = 1e-6,
		reltol = 1e-6,
		maxiters = 1e7,
	)

	return sol
end

function compute_peaks(sol, t_events, v)
	u = sol(sol.t; idxs = v).u
	[maximum(u[findall(t -> s <= t <= e, sol.t)]) for (s, e) in t_events]
end

function solve_system_learning(sys, tspan; delay, duration, αs, γs)
	cbs, evs = generate_callbacks_learning(sys, tspan, delay, duration)

	ssprob = SteadyStateProblem(sys, [])

	prob = ODEProblem(sys, [], tspan;
		callback = cbs,
	)

	sol = let
		α = 1.0
		γ = 1e-2
		solve_problem_learning(prob, ssprob, 1.0, 1e-2)
	end

	t_range = collect(range(extrema(sol.t)..., 10_000))
	uks = map(unknowns(sys)) do uk
		s = Symbol("sample/"*string(uk))
		u = sol(t_range; idxs = uk).u
		(s, u)
	end

	sample = vcat((Symbol("sample/t"), t_range), uks...)

	ps = let
		p = parameters(sys)
		sym_p = Symbol.("parameters/".*string.(p))
		collect(zip(sym_p, prob.p[1]))
	end

	peaks = map(Iterators.product(αs, γs)) do (α, γ)
		sol = solve_problem_learning(prob, ssprob, α, γ)
		compute_peaks(sol, evs, sys.G)
	end

	return Dict(
		:α_list => αs,
		:γ_list => γs,
		:peaks => peaks,
		:tspan => tspan,
		:event_delay => delay,
		:event_duration => duration,
		:equations => string.(equations(sys)),
		:parameters => ps,
		:sample => sample,
	)
end

tspan = (0, 150)
shared_learning_args = Dict(
	:delay => 10.0,
	:duration => 1.0,
	:αs => 10 .^ (range(log10(1e-2), log10(1e2), 1000)),
	:γs => 10 .^ (range(log10(5e-3), log10(1/2), 1000)),
)

# Habituation

function habituation_system()
	@parameters begin
		α = 1.0
		β = 5.0
		μ = 1.0
		γ = 1e-2
		λ = 1.0
	end

	@variables begin
		x(t) = 0.0
		X(t) = 0.0
		I(t) = 0.0
		G(t) = 0.0
	end

	eqs = [
		D(x) ~ 0,
		D(X) ~ μ - λ*X,
		D(I) ~ α*θp(X*x, 1) - γ*I,
		D(G) ~ β*θp(X*x, 1)*θm(I, 1) - λ*G,
	]

	@mtkcompile sys = System(eqs, t)
end

habituation_data = let
	sys = habituation_system()
	solve_system_learning(sys, tspan; shared_learning_args...)
end

jldsave("habituation_heatmap.jld2";
	equations = habituation_data[:equations],
	event_delay = habituation_data[:event_delay],
	event_duration = habituation_data[:event_duration],
	fc = (p -> log2(p[end]/p[1])).(habituation_data[:peaks]),
	peaks = habituation_data[:peaks],
	tspan = habituation_data[:tspan],
	α_list = habituation_data[:α_list],
	γ_list = habituation_data[:γ_list],
	habituation_data[:parameters]...,
	habituation_data[:sample]...,
)

# Sensitization

function sensitization_system()
	@parameters begin
		α = 1.0
		β = 5.0
		μ = 1.0
		γ = 1e-2
		λ = 1.0
		ρ = 1.5
	end

	@variables begin
		x(t) = 0.0
		X(t) = 0.0
		I(t) = 0.0
		R(t) = 0.0
		G(t) = 0.0
	end

	eqs = [
		D(x) ~ 0,
		D(X) ~ μ*θm(R, 1) - λ*X,
		D(I) ~ α*θp(X*x, 1) - γ*I,
		D(R) ~ ρ*θm(I, 1) - λ*R,
		D(G) ~ β*θp(X*x, 1) - λ*G,
	]

	@mtkcompile sys = System(eqs, t)
end

sensitization_data = let
	sys = sensitization_system()
	solve_system_learning(sys, tspan; shared_learning_args...)
end

jldsave("sensitization_heatmap.jld2";
	equations = sensitization_data[:equations],
	event_delay = sensitization_data[:event_delay],
	event_duration = sensitization_data[:event_duration],
	fc = (p -> log2(p[end]/p[1])).(sensitization_data[:peaks]),
	peaks = sensitization_data[:peaks],
	tspan = sensitization_data[:tspan],
	α_list = sensitization_data[:α_list],
	γ_list = sensitization_data[:γ_list],
	sensitization_data[:parameters]...,
	sensitization_data[:sample]...,
)

# Combining Sensitization and Habituation

function hybrid_system()
	@parameters begin
		α = 1.0
		β = 5.0
		μ = 1.0
		γ = 1e-2
		λ = 1.0
		ρ = 1.75
	end

	@variables begin
		x(t) = 0.0
		X(t) = 0.0
		I(t) = 0.0
		R(t) = 0.0
		G(t) = 0.0
	end

	eqs = [
		D(x) ~ 0,
		D(X) ~ μ*θm(R, 1) - λ*X,
		D(I) ~ α*θp(X*x, 1) - γ*I,
		D(R) ~ ρ*θm(I, 1) - λ*R,
		D(G) ~ β*θp(X*x, 1)*θm(I, 2) - λ*G,
	]

	@mtkcompile sys = System(eqs, t)
end

hybrid_data = let
	sys = hybrid_system()
	solve_system_learning(sys, 3 .* tspan; shared_learning_args...)
end

jldsave("hybrid_heatmap.jld2";
	equations = hybrid_data[:equations],
	event_delay = hybrid_data[:event_delay],
	event_duration = hybrid_data[:event_duration],
	fc_habituation = (p -> log2(p[end]/maximum(p))).(hybrid_data[:peaks]),
	fc_sensitization = (p -> log2(maximum(p)/p[1])).(hybrid_data[:peaks]),
	peaks = hybrid_data[:peaks],
	tspan = hybrid_data[:tspan],
	α_list = hybrid_data[:α_list],
	γ_list = hybrid_data[:γ_list],
	hybrid_data[:parameters]...,
	hybrid_data[:sample]...,
)

# Massed--Spaced
function massed_system()
	@parameters begin
		α = 1.0
		β = 1.0
		γ = 1e-2
		λ = 1.0
		μ = 1.0
	end
	
	@variables begin
		G(t) = 0.0
		X(t) = 0.0
		x(t) = 0.0
		A(t) = 0.0
	end
	
	eqs = [
		D(x) ~ 0,
		D(X) ~ μ - λ*X,
		D(A) ~ α*θp(X*x, 1.0) - γ*A,
		D(G) ~ β*θp(A, 1.0) - γ*G,
	]

	@mtkcompile sys = System(eqs, t)
end

function solve_massed(sys, tspan, repeats, delay, duration)
	duration_single = duration/repeats
	t_start_events = if delay == 0.0
		first(tspan) + 1
	else
		step = delay + duration_single
		(first(tspan) + 1:step:last(tspan))[1:repeats]
	end
	t_end_events = if length(t_start_events) == 1
		t_start_events .+ duration
	else
		t_start_events .+ duration_single
	end

	# Find the index of x in the state variables
	x_pos = findfirst(isequal(sys.x), unknowns(sys))

	# Find the index of G in the state variables
	G_pos = findfirst(isequal(sys.G), unknowns(sys))

	# Find extrema.
	dG_zero = (u, t, integrator) -> begin
		du = similar(u)
		integrator.f(du, u, integrator.p, t)
		du[G_pos]
	end
	
	cbs = CallbackSet(
		PositiveDomain(save = false),
		TerminateSteadyState(min_t = last(t_end_events) + delay),
		PresetTimeCallback(
			t_start_events,
			(integrator) -> integrator.u[x_pos] = 1,
		),
		PresetTimeCallback(
			t_end_events,
			(integrator) -> integrator.u[x_pos] = 0,
		),
		# Find maxima.
		ContinuousCallback(dG_zero, nothing, i -> nothing),
	)

	# Solve steady state problem
	prob_ss = SteadyStateProblem(sys, [])
	sol_ss = solve(prob_ss, DynamicSS(Rodas5P());
		abstol = 1e-8,
		reltol = 1e-8,
	)

	# Solve ODE problem
	u_ss = [v => u for (v, u) in zip(unknowns(sys), sol_ss.u)]
	ode = ODEProblem(sys, u_ss, tspan; callback = cbs)
	sol = solve(ode, AutoTsit5(Rosenbrock23());
		abstol = 1e-6,
		reltol = 1e-6,
		maxiters = 1e7,
	)

	(sol, collect(zip(t_start_events, t_end_events)))
end

tspan = (0.0, 1e9)

duration = 10.0
repeats = 1:1_000
delays = 10 .^ (range(log10(1e-2), log10(1e4), 1000))

massed_spaced_peaks = map(Iterators.product(repeats, delays)) do (r, d)
	sys = massed_system()
	sol, _ = solve_massed(sys, tspan, r, d, duration)
	maximum(sol(sol.t; idxs = sys.G))
end

massed_peak = let
	sys = massed_system()
	sol, _ = solve_massed(sys, tspan, 1, 0.0, duration)
	maximum(sol(sol.t; idxs = sys.G))
end

massed_parameters = let
	sys = massed_system()
	sym_params = Symbol.("parameters/".*string.(parameters(sys)))
	p = ODEProblem(sys, [], tspan).p
	collect(zip(sym_params, p[1]))
end

jldsave("massed_heatmap.jld2";
	delays,
	duration,
	equations = string.(equations(massed_system())),
	fc = log2.(massed_spaced_peaks ./ massed_peak),
	massed_peak,
	peaks = massed_spaced_peaks,
	repeats,
	tspan,
	massed_parameters...,
)
