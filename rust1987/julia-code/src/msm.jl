# ==========================================================================
# msm.jl — Method of Simulated Moments criterion
#
# Julia translation of msm.objective (msm.estim was a TODO stub upstream
# and is not ported). Moments = fractions of the fleet in each mileage bin.
# ==========================================================================

"""
    msm_moments(m; use_solution=true, nsims=1000, Tsims=1000, scale=100.0,
                  rng, v0, opts) -> (sim_moments, sim_moments_indiv)

Model-implied mileage-distribution moments at parameters `m` (n-1 of the n
bin fractions — the last is dropped because they sum to one, scaled by
`scale`). With `use_solution=true` the exact stationary distribution is
used; otherwise moments are simulated with `nsims` buses over `Tsims`
months, and `sim_moments_indiv` holds per-bus moments (for the optimal
weighting matrix).
"""
function msm_moments(m::Model; use_solution::Bool=true, nsims::Int=1000,
                     Tsims::Int=1000, scale::Float64=100.0,
                     rng::AbstractRNG=Xoshiro(12345),
                     v0::Vector{Float64}=zeros(m.n),
                     opts::SolverOptions=SolverOptions())
    v = copy(v0); pk = zeros(m.n)
    solve_poly!(v, pk, EVBellman(m); opts=opts)

    if use_solution
        pp, _, _ = eqb(m, pk)
        return pp[1:end-1] .* scale, nothing
    end

    sim = simdata(nsims, Tsims, m, pk; rng=rng)
    counts = zeros(Int, m.n)
    indiv = zeros(nsims, m.n)
    for i in eachindex(sim.x)
        counts[sim.x[i]] += 1
        indiv[sim.id[i], sim.x[i]] += 1
    end
    sim_moments = counts[1:end-1] ./ (nsims * Tsims) .* scale
    sim_moments_indiv = indiv[:, 1:end-1] ./ Tsims .* scale
    return sim_moments, sim_moments_indiv
end

"""
    msm_objective(data_moments, W, m; kwargs...) -> Float64

Quadratic-form MSM criterion `(m̂(θ) − m̄)′ W (m̂(θ) − m̄)` where `m̂(θ)` are
the model moments at `m` (see `msm_moments`) and `m̄` the data moments.
"""
function msm_objective(data_moments::Vector{Float64}, W::AbstractMatrix{Float64},
                       m::Model; kwargs...)
    sim_moments, _ = msm_moments(m; kwargs...)
    mom = sim_moments .- data_moments
    return dot(mom, W * mom)
end
