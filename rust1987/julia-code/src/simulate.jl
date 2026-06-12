# ==========================================================================
# simulate.jl — Simulation and equilibrium distribution
#
# Julia translation of zurcher.simdata and zurcher.eqb
# ==========================================================================

"""
    simdata(N, T, m, pk; rng=Random.default_rng()) -> NamedTuple

Simulate `N` buses for `T` months from the model `m` under the keep
probabilities `pk` (translation of `zurcher.simdata`).

Initial mileage is uniform on the state space. Returns column vectors of
length N·T: `(id, t, x, x1, d, dx1)` where `d` is the replacement indicator
and `x1` the next-period state `min(x·(1-d) + d + dx1, n)`.
"""
function simdata(N::Int, T::Int, m::Model, pk::Vector{Float64};
                 rng::AbstractRNG=Random.default_rng())
    n = m.n
    cpf = cumsum(pfull(m))

    id = repeat(1:N, inner=T)
    t = repeat(1:T, outer=N)
    x = Vector{Int}(undef, N * T)
    x1 = Vector{Int}(undef, N * T)
    d = Vector{Bool}(undef, N * T)
    dx1 = Vector{Int}(undef, N * T)

    for b in 1:N
        xt = rand(rng, 1:n)                       # uniform initial condition
        for τ in 1:T
            i = (b - 1) * T + τ
            dx = searchsortedfirst(cpf, rand(rng)) - 1    # mileage increment
            dt = rand(rng) < 1.0 - pk[xt]                 # replace?
            x[i] = xt
            d[i] = dt
            dx1[i] = dx
            x1[i] = min((dt ? 1 : xt) + dx, n)            # reset to 1 if replaced
            xt = x1[i]
        end
    end
    return (id=id, t=t, x=x, x1=x1, d=d, dx1=dx1)
end

"Convert simulated columns to the estimation sample type."
BusData(sim::NamedTuple) = BusData(sim.x, sim.d, sim.dx1)

"""
    ergodic(Π) -> Vector

Stationary distribution of the Markov matrix `Π` (rows sum to 1): solves
`π′Π = π′, Σπ = 1` as the least-squares solution of `[(I − Π′); 1′]π = [0; 1]`.
"""
function ergodic(Π::AbstractMatrix{Float64})
    n = size(Π, 1)
    A = [Matrix(I - Π'); ones(1, n)]
    b = [zeros(n); 1.0]
    return A \ b
end

"""
    eqb(m, pk) -> (pp, pp_K, pp_R)

Equilibrium distribution of the controlled mileage process (translation of
`zurcher.eqb`): `pp[x] = Pr{x}`, `pp_K[x] = Pr{x, keep}`, `pp_R[x] = Pr{x, replace}`.
"""
function eqb(m::Model, pk::Vector{Float64})
    Π = policy_matrix(m, pk)         # P_keep .* pk + P_replace .* (1-pk)
    pp = ergodic(Matrix(Π))
    return pp, pp .* pk, pp .* (1.0 .- pk)
end

"""
    equilibrium_demand(m, RCgrid; v0=zeros(m.n), opts) -> Vector

Expected annual engine replacement demand as a function of the replacement
cost (the inner loop of run_dem.m): for each RC, re-solve the model, compute
the equilibrium distribution, and return `12·Σ_x Pr{x, replace}`.
"""
function equilibrium_demand(m::Model, RCgrid::AbstractVector{<:Real};
                            v0::Vector{Float64}=zeros(m.n),
                            opts::SolverOptions=SolverOptions())
    demand = similar(RCgrid, Float64)
    v = copy(v0)
    pk = zeros(m.n)
    for (i, rc) in enumerate(RCgrid)
        mi = update(m; RC=float(rc))
        solve_poly!(v, pk, EVBellman(mi); opts=opts)     # warm-started in RC
        _, _, pp_R = eqb(mi, pk)
        demand[i] = 12.0 * sum(pp_R)
    end
    return demand
end
