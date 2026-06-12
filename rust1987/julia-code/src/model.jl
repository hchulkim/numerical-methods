# ==========================================================================
# model.jl — Primitives of Rust's (1987) bus engine replacement model
#
# Julia translation of matlab-code/zurcher.m
#
# State:   x ∈ {1, …, n}  — discretized odometer reading (grid value x-1)
# Choice:  keep (pay maintenance cost) or replace (pay RC, reset to x = 1)
# Mileage: moves up by j-1 grid cells with probability π_j, j = 1, …, J,
#          clamped at the top cell n (absorbing boundary).
# ==========================================================================

"""
    Model(; n=175, maxmiles=450.0, RC=11.7257, c=2.45569, β=0.9999,
            p=[0.0937, 0.4475, 0.4459, 0.0127])

Primitives of Harold Zurcher's engine replacement problem.

Fields
- `n`        : number of mileage grid points
- `maxmiles` : maximum mileage (in 1000s of miles) covered by the grid
- `RC`       : engine replacement cost
- `c`        : maintenance cost slope (cost at state x is `0.001*c*(x-1)`)
- `β`        : discount factor
- `p`        : *free* mileage-increment probabilities; the last increment
               probability is implied as `1 - sum(p)`

Defaults are the Table X estimates of Rust (1987), exactly as in
`zurcher.setup` of the MATLAB code.
"""
Base.@kwdef struct Model
    n::Int = 175
    maxmiles::Float64 = 450.0
    RC::Float64 = 11.7257
    c::Float64 = 2.45569
    β::Float64 = 0.9999
    p::Vector{Float64} = [0.0937, 0.4475, 0.4459, 0.0127]
end

"Copy `m` with selected parameters replaced (Model is immutable)."
update(m::Model; RC=m.RC, c=m.c, β=m.β, p=m.p, n=m.n, maxmiles=m.maxmiles) =
    Model(n=n, maxmiles=maxmiles, RC=RC, c=c, β=β, p=p)

"Full mileage-increment distribution `[p; 1 - sum(p)]`."
pfull(m::Model) = [m.p; 1.0 - sum(m.p)]

"Mileage grid `0, 1, …, n-1` (index x ↦ grid value x-1)."
mileage_grid(m::Model) = collect(0.0:m.n-1)

"Maintenance cost at state index `x`: `0.001*c*(x-1)`."
cost(m::Model, x::Integer) = 0.001 * m.c * (x - 1)

"Flow utility of keeping at state `x`."
u_keep(m::Model, x::Integer) = -cost(m, x)

"Flow utility of replacing (state-independent; `cost(m,1) == 0` kept for fidelity)."
u_replace(m::Model) = -m.RC - cost(m, 1)

# --------------------------------------------------------------------------
# Transition structure
#
# The MATLAB code stores the keep-transition as a sparse n×n matrix P{1} and
# multiplies it repeatedly. Both transition matrices have so much structure
# that we never need to build them in the hot loops:
#
#   * keep:    row x has at most J nonzeros at columns min(x+j-1, n), j=1:J,
#              so P_keep*V is a banded O(n·J) loop (`keep_mul!`),
#   * replace: every row equals (π_1, …, π_J, 0, …, 0), so P_repl*V is the
#              same scalar for every state (`replace_expectation`).
# --------------------------------------------------------------------------

"""
    keep_mul!(out, pf, V) -> out

In-place `out = P_keep * V` where `P_keep` is the mileage transition matrix
conditional on keeping: from state `x` move to `min(x+j-1, n)` w.p. `pf[j]`.
Clamping at `n` reproduces the absorbing top cell of the MATLAB sparse
construction (all mass that would overflow piles up on `V[n]`).
"""
function keep_mul!(out::AbstractVector{Float64}, pf::Vector{Float64}, V::AbstractVector{Float64})
    n = length(V)
    @inbounds for x in 1:n
        s = 0.0
        for j in eachindex(pf)
            s += pf[j] * V[min(x + j - 1, n)]
        end
        out[x] = s
    end
    return out
end

"""
    replace_expectation(pf, V)

`E[V(x′) | replace]`: replacement resets the odometer, so `x′ = j` w.p.
`pf[j]` — identical for every current state (one dot product, not a matvec).
"""
function replace_expectation(pf::Vector{Float64}, V::AbstractVector{Float64})
    s = 0.0
    @inbounds for j in eachindex(pf)
        s += pf[j] * V[min(j, length(V))]
    end
    return s
end

"""
    statetransition(m) -> (P_keep, P_replace)

Explicit sparse transition matrices, mirroring `zurcher.statetransition`.
Only used where a full matrix is genuinely needed (equilibrium distribution,
inspection); the solver and likelihood never build them.
"""
function statetransition(m::Model)
    pf = pfull(m); n = m.n; J = length(pf)
    rows = Int[]; cols = Int[]; vals = Float64[]
    sizehint!(rows, n * J); sizehint!(cols, n * J); sizehint!(vals, n * J)
    for x in 1:n, j in 1:J
        push!(rows, x); push!(cols, min(x + j - 1, n)); push!(vals, pf[j])
    end
    P_keep = sparse(rows, cols, vals, n, n)             # duplicates at col n are summed
    P_replace = sparse(repeat(1:n, inner=J), repeat(1:J, outer=n),
                       repeat(pf, outer=n), n, n)
    return P_keep, P_replace
end

# --------------------------------------------------------------------------
# Bellman operators
#
# Two equivalent formulations (as in the MATLAB code, mp.bellman_type):
#   EV space (Rust's original): fixed point is ev(x) = E[V(x′) | x, keep]
#   IV space:                   fixed point is the integrated value V(x)
# Both recenter the log-sum-exp to avoid numerical overflow.
# --------------------------------------------------------------------------

abstract type BellmanOperator end

"""
    EVBellman(m)

Bellman operator Γ in *expected value* space (Rust's NFXP manual):

    ev′(x) = Σ_j π_j · V(min(x+j-1, n)),
    V(y)   = logsumexp(vK(y), vR),
    vK(y)  = u_keep(y) + β·ev(y),      vR = u_replace + β·ev(1).

Holds preallocated buffers so repeated applications allocate nothing.
After `apply!`, the buffer `Γ.V` contains the integrated values at the
evaluation point (needed by the likelihood derivatives).
"""
struct EVBellman <: BellmanOperator
    m::Model
    pf::Vector{Float64}
    V::Vector{Float64}      # logsumexp (integrated) values, refreshed by apply!
end
EVBellman(m::Model) = EVBellman(m, pfull(m), zeros(m.n))

"""
    apply!(out, pk, Γ, v) -> out

One application of the Bellman operator: `out = Γ(v)`. Also fills `pk` with
the keep-probabilities `P(keep | x)` implied by `v`.
"""
function apply!(out::Vector{Float64}, pk::Vector{Float64}, Γ::EVBellman, ev::Vector{Float64})
    m = Γ.m; β = m.β
    vR = u_replace(m) + β * ev[1]
    @inbounds for x in 1:m.n
        vK = u_keep(m, x) + β * ev[x]
        mx = max(vK, vR)                                  # recenter: no overflow
        Γ.V[x] = mx + log(exp(vK - mx) + exp(vR - mx))
        pk[x] = 1.0 / (1.0 + exp(vR - vK))
    end
    keep_mul!(out, Γ.pf, Γ.V)
    return out
end

"""
    frechet(Γ, pk) -> SparseMatrixCSC

Fréchet derivative ∂Γ/∂ev evaluated where keep-probabilities are `pk`:

    ∂ev′(x)/∂ev(y) = β·P_keep(x,y)·pk(y)            for y ≥ 2,
    ∂ev′(x)/∂ev(1) = β·P_keep(x,1)·pk(1) + β·Σ_y P_keep(x,y)·(1-pk(y)),

the second term because ev(1) enters the replacement value at every state.
Matches `dbellman_dev` in `zurcher.bellman_ev`.
"""
function frechet(Γ::EVBellman, pk::Vector{Float64})
    m = Γ.m; n = m.n; β = m.β; pf = Γ.pf; J = length(pf)
    nnz_max = n * (J + 1)
    rows = Vector{Int}(undef, nnz_max); cols = Vector{Int}(undef, nnz_max)
    vals = Vector{Float64}(undef, nnz_max)
    k = 0
    @inbounds for x in 1:n
        extra1 = 0.0                                      # ∂/∂ev(1) via replacement
        for j in 1:J
            y = min(x + j - 1, n)
            k += 1; rows[k] = x; cols[k] = y; vals[k] = β * pf[j] * pk[y]
            extra1 += β * pf[j] * (1.0 - pk[y])
        end
        k += 1; rows[k] = x; cols[k] = 1; vals[k] = extra1
    end
    return sparse(rows, cols, vals, n, n)                 # duplicate entries are summed
end

"""
    IVBellman(m)

Bellman operator in *integrated value* space:

    V′(x) = logsumexp(vK(x), vR),
    vK(x) = u_keep(x) + β·(P_keep V)(x),   vR = u_replace + β·(P_repl V)(x).

Note `(P_repl V)(x)` is the same scalar for every `x`.
"""
struct IVBellman <: BellmanOperator
    m::Model
    pf::Vector{Float64}
    evK::Vector{Float64}    # buffer: E[V′ | x, keep]
end
IVBellman(m::Model) = IVBellman(m, pfull(m), zeros(m.n))

function apply!(out::Vector{Float64}, pk::Vector{Float64}, Γ::IVBellman, V::Vector{Float64})
    m = Γ.m; β = m.β
    keep_mul!(Γ.evK, Γ.pf, V)
    evR = replace_expectation(Γ.pf, V)
    vR = u_replace(m) + β * evR
    @inbounds for x in 1:m.n
        vK = u_keep(m, x) + β * Γ.evK[x]
        mx = max(vK, vR)
        out[x] = mx + log(exp(vK - mx) + exp(vR - mx))
        pk[x] = 1.0 / (1.0 + exp(vR - vK))
    end
    return out
end

"Fréchet derivative in IV space: `β·(P_keep .* pk + P_repl .* (1-pk))` (row-wise weights)."
function frechet(Γ::IVBellman, pk::Vector{Float64})
    m = Γ.m; n = m.n; β = m.β; pf = Γ.pf; J = length(pf)
    nnz_max = 2 * n * J
    rows = Vector{Int}(undef, nnz_max); cols = Vector{Int}(undef, nnz_max)
    vals = Vector{Float64}(undef, nnz_max)
    k = 0
    @inbounds for x in 1:n, j in 1:J
        k += 1; rows[k] = x; cols[k] = min(x + j - 1, n); vals[k] = β * pf[j] * pk[x]
        k += 1; rows[k] = x; cols[k] = min(j, n);         vals[k] = β * pf[j] * (1.0 - pk[x])
    end
    return sparse(rows, cols, vals, n, n)
end

"Convert a fixed point in IV space to EV space: `ev = P_keep * V`."
function iv_to_ev(Γ::IVBellman, V::Vector{Float64})
    ev = similar(V)
    keep_mul!(ev, Γ.pf, V)
    return ev
end
