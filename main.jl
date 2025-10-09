using DiffEqCallbacks
using JLD2
using ModelingToolkit
using OrdinaryDiffEq
using SteadyStateDiffEq

using ModelingToolkit: t_nounits as t, D_nounits as D

# Define the helper functions
θp(L, K) = L^2/(K + L^2)
θm(L, K) = 1/(1 + (L/K)^2)

function generate_callbacks(model, tspan, event_delay, event_duration)
	# Create event time points
	t_start_events = (tspan[1]:event_delay:tspan[2])[2:end-1]
	t_end_events = t_start_events .+ event_duration
	x_var = ModelingToolkit.getvar(model, :x)
	x_pos = findfirst(isequal(x_var), unknowns(model))
	
	# Create callbacks for events
	cbs = CallbackSet(
		PositiveDomain(save = false),
		PresetTimeCallback(
			t_start_events,
			(integrator) -> integrator.u[x_pos] = 1
		),
		PresetTimeCallback(
			t_end_events,
			(integrator) -> integrator.u[x_pos] = 0
		)
	)
	events = collect(zip(t_start_events, t_end_events))
	return (cbs, events)
end

function solve_problem_learning(def_prob, def_ssprob, α, δ)
	ssprob = remake(def_ssprob; p = [:α => α, :δ => δ])
	sssol = solve(ssprob, DynamicSS(Rodas5P()))
	prob = remake(def_prob; u0 = sssol.u, p = [:α => α, :δ => δ])
	sol = solve(prob, AutoTsit5(Rosenbrock23()); maxiters = 1e7)
end

function compute_peaks(sol, t_events, v)
	u = sol(sol.t; idxs = v).u
	[maximum(u[findall(t -> s <= t <= e, sol.t)]) for (s, e) in t_events]
end

tspan = (0, 1_500)
event_delay = 100.0
event_duration = 10.0

α_list = 10 .^ (range(log10(1e-3), log10(1e1), 400))
δ_list = 10 .^ (range(log10(1e-5), log10(1e0), 400))

αδ_matrix = collect(Iterators.product(α_list, δ_list))

# Habituation

@mtkmodel Habituation begin
	@parameters begin
		α = 1e-1
		β = 2.0
		γ = 1.0
		δ = 1e-3
		λ = 1.0
	end
	@variables begin
		x(t) = 0.0
		X(t) = 0.0
		I(t) = 0.0
		G(t) = 0.0
	end
	@equations begin
		D(x) ~ 0
		D(X) ~ γ - λ*X
		D(I) ~ α*θp(X*x, 1) - δ*I
		D(G) ~ β*θp(X*x, 1)*θm(I, 1) - λ*G
	end
end

@mtkbuild habituation = Habituation()

habituation_cbs, habituation_ev = begin
	generate_callbacks(habituation, tspan, event_delay, event_duration)
end
habituation_ssprob = SteadyStateProblem(habituation, [])
habituation_prob = ODEProblem(habituation, [], tspan;
	callback = habituation_cbs,
)

habituation_sol = begin
	α = 1e-1
	δ = 1e-3
	solve_problem_learning(habituation_prob, habituation_ssprob, α, δ)
end

habituation_peaks = Matrix{Vector{Float64}}(undef, size(αδ_matrix)...)
for (i, (α, δ)) in collect(enumerate(αδ_matrix))
	sol = solve_problem_learning(habituation_prob, habituation_ssprob, α, δ)
	peaks = compute_peaks(sol, habituation_ev, habituation.G)
	habituation_peaks[i] = clamp.(peaks, 0, maximum(peaks))
end

habituation_fc = (p -> log2(p[end]/p[1])).(habituation_peaks)

habituation_parameters = begin
	sym_params = Symbol.("parameters/".*string.(parameters(habituation)))
	collect(zip(sym_params, habituation_prob.p[1]))
end

jldsave("habituation_heatmap.jld2";
	α_list,
	δ_list,
	peaks = habituation_peaks,
	fc = habituation_fc,
	tspan,
	event_delay,
	event_duration,
	equations = string.(equations(habituation)),
	habituation_parameters...,
)

# Sensitization

@mtkmodel Sensitization begin
	@parameters begin
		α = 1e-1
		β = 2.0
		γ = 1.0
		δ = 1e-3
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
	@equations begin
		D(x) ~ 0
		D(X) ~ γ*θm(R, 1) - λ*X
		D(I) ~ α*θp(X*x, 1) - δ*I
		D(R) ~ ρ*θm(I, 1) - λ*R
		D(G) ~ β*θp(X*x, 1) - λ*G
	end
end

@mtkbuild sensitization = Sensitization()

sensitization_cbs, sensitization_ev = begin
	generate_callbacks(sensitization, tspan, event_delay, event_duration)
end

sensitization_ssprob = SteadyStateProblem(sensitization, [])
sensitization_prob = ODEProblem(sensitization, [], tspan;
	callback = sensitization_cbs,
)

sensitization_sol = begin
	α = 1e-1
	δ = 1e-3
	solve_problem_learning(sensitization_prob, sensitization_ssprob, α, δ)
end

sensitization_peaks = Matrix{Vector{Float64}}(undef, size(αδ_matrix)...)
for (i, (α, δ)) in collect(enumerate(αδ_matrix))
	sol = solve_problem_learning(sensitization_prob, sensitization_ssprob, α, δ)
	peaks = compute_peaks(sol, sensitization_ev, sensitization.G)
	sensitization_peaks[i] = clamp.(peaks, 0, maximum(peaks))
end

sensitization_fc = (p -> log2(p[end]/p[1])).(sensitization_peaks)

sensitization_parameters = begin
	sym_params = Symbol.("parameters/".*string.(parameters(sensitization)))
	collect(zip(sym_params, sensitization_prob.p[1]))
end

jldsave("sensitization_heatmap.jld2";
	α_list,
	δ_list,
	peaks = sensitization_peaks,
	fc = sensitization_fc,
	tspan,
	event_delay,
	event_duration,
	equations = string.(equations(sensitization)),
	sensitization_parameters...,
)

# Combining Sensitization and Habituation

@mtkmodel Hybrid begin
	@parameters begin
		α = 1e-1
		β = 4.0
		γ = 1.0
		δ = 1e-3
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
	@equations begin
		D(x) ~ 0
		D(X) ~ γ*θm(R, 1) - λ*X
		D(I) ~ α*θp(X*x, 1) - δ*I
		D(R) ~ ρ*θm(I, 1) - λ*R
		D(G) ~ β*θp(X*x, 1)*θm(I, 2) - λ*G
	end
end

@mtkbuild hybrid = Hybrid()

hybrid_cbs, hybrid_ev = begin
	generate_callbacks(hybrid, 2 .* tspan, event_delay, event_duration)
end

hybrid_ssprob = SteadyStateProblem(hybrid, [])
hybrid_prob = ODEProblem(hybrid, [], 2 .* tspan;
	callback = hybrid_cbs,
)

hybrid_sol = begin
	α = 1e-1
	δ = 1e-3
	solve_problem_learning(hybrid_prob, hybrid_ssprob, α, δ)
end

hybrid_peaks = Matrix{Vector{Float64}}(undef, size(αδ_matrix)...)
for (i, (α, δ)) in collect(enumerate(αδ_matrix))
	sol = solve_problem_learning(hybrid_prob, hybrid_ssprob, α, δ)
	peaks = compute_peaks(sol, hybrid_ev, hybrid.G)
	hybrid_peaks[i] = clamp.(peaks, 0, maximum(peaks))
end

hybrid_fc_habituation  = (p -> log2(p[end]/maximum(p))).(hybrid_peaks)
hybrid_fc_sensitization = (p -> log2(maximum(p)/p[1])).(hybrid_peaks)

hybrid_parameters = begin
	sym_params = Symbol.("parameters/".*string.(parameters(hybrid)))
	collect(zip(sym_params, hybrid_prob.p[1]))
end

jldsave("hybrid_heatmap.jld2";
	α_list,
	δ_list,
	peaks = hybrid_peaks,
	fc_hab = hybrid_fc_habituation,
	fc_sens = hybrid_fc_sensitization,
	tspan,
	event_delay,
	event_duration,
	equations = string.(equations(hybrid)),
	hybrid_parameters...,
)

# Massed--Spaced

@mtkmodel Massed begin
	@parameters begin
		α = 1e-1
		β = 1e-1
		δ = 1e-3
		λ = 1e+0
		γ = 1e+1
	end
	
	@variables begin
		G(t) = 0.0
		X(t) = 0.0
		x(t) = 0.0
		A(t) = 0.0
	end
	
	@equations begin
		D(x) ~ 0
		D(X) ~ γ - λ*X
		D(A) ~ α*θp(X*x, 1.0) - δ*A
		D(G) ~ β*θp(A, 1.0) - δ*G
	end
end

@mtkbuild massed = Massed()

function solve_massed(sys, tspan, ev_repeats, ev_delay, ev_duration)
	duration_single = ev_duration/ev_repeats
	t_start_events = if ev_delay == 0.0
		tspan[1] + 1
	else
		step = ev_delay + duration_single
		(tspan[1] + 1:step:tspan[2])[1:ev_repeats]
	end
	t_end_events = if length(t_start_events) == 1
		t_start_events .+ ev_duration
	else
		t_start_events .+ duration_single
	end
	
	# Find the index of x in the state variables
	find_x_pos = findfirst(isequal(sys.x), unknowns(sys))
	
	cbs = CallbackSet(
		PositiveDomain(save = false),
		TerminateSteadyState(min_t = last(t_end_events)),
		PresetTimeCallback(
			t_start_events,
			(i) -> i.u[find_x_pos] = 1,
		),
		PresetTimeCallback(
			t_end_events,
			(i) -> i.u[find_x_pos] = 0,
		),
	)
	
	# Solve steady state problem
	ssprob = SteadyStateProblem(sys, [])
	sssol = solve(ssprob, DynamicSS(Rodas5P()))
	
	# Solve ODE problem
	ode = ODEProblem(sys, sssol.u, tspan; callback = cbs)
	sol = solve(ode, AutoTsit5(Rosenbrock23());
		maxiters = 1e7,
	)
	
	events = collect(zip(t_start_events, t_end_events))
	(sol, events)
end

tspan = (0.0, 1e9)

ev_duration = 100.0
ev_repeats = unique(floor.(Int64, 10 .^ (range(log10(1), log10(1e4), 500))))
ev_delays = 10 .^ (range(log10(1e-2), log10(1e5), 500))

delay_repeat_matrix = collect(Iterators.product(ev_delays, ev_repeats))

massed_spaced_peaks = Matrix{Float64}(undef, size(delay_repeat_matrix)...)
for (i, (d, r)) in collect(enumerate(delay_repeat_matrix))
	sol, _ = solve_massed(massed, tspan, r, d, ev_duration)
	massed_spaced_peaks[i] = maximum(sol(sol.t; idxs = massed.G))
end

massed_peak = begin
	sol, events = solve_massed(massed, tspan, 1, 0.0, ev_duration)
	maximum(sol(sol.t; idxs = massed.G))
end

massed_parameters = begin
	sym_params = Symbol.("parameters/".*string.(parameters(massed)))
	p = ODEProblem(massed, [], tspan).p
	collect(zip(sym_params, p[1]))
end

jldsave("massed_heatmap.jld2";
	ev_repeats,
	ev_delays,
	peaks = massed_spaced_peaks,
	massed_peak,
	tspan,
	ev_duration,
	equations = string.(equations(massed)),
	massed_parameters...,
)
