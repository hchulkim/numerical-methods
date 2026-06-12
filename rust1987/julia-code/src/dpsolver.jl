# ==========================================================================
# dpsolver.jl — Fixed-point solver for the Bellman operator
#
# Julia translation of matlab-code/dpsolver.m
#
# The "poly-algorithm" of Rust's NFXP manual: start with cheap successive
# approximations (SA, global convergence, linear rate β), and switch to
# Newton–Kantorovich (NK, quadratic rate) once the observed contraction
# rate stabilizes near β — i.e. once SA has entered the domain of
# attraction of Newton's method.
# ==========================================================================

"""
    SolverOptions(; kwargs...)

Algorithm parameters (defaults = `dpsolver.setup` in MATLAB).

- `sa_max`      : max successive-approximation steps per stage (50)
- `sa_min`      : min successive-approximation steps before any switch (10)
- `sa_tol`      : SA convergence tolerance, scale-adjusted (1e-10)
- `max_fxpiter` : max alternations between SA and NK stages (5)
- `nk_max`      : max Newton–Kantorovich steps per stage (10)
- `nk_tol`      : final NK tolerance, scale-adjusted (1e-12)
- `tol_ratio`   : switch SA→NK when `|β - tol_i/tol_{i-1}| < tol_ratio` (1e-2)
- `verbose`     : 0 silent, 1 summary, 2 per-iteration detail
"""
Base.@kwdef struct SolverOptions
    sa_max::Int = 50
    sa_min::Int = 10
    sa_tol::Float64 = 1e-10
    max_fxpiter::Int = 5
    nk_max::Int = 10
    nk_tol::Float64 = 1e-12
    tol_ratio::Float64 = 1e-2
    verbose::Int = 0
end

"Iteration record returned by the solver (used e.g. by run_errorbound.jl)."
mutable struct SolverInfo
    converged::Bool
    sa_tols::Vector{Float64}    # SA error bounds, in order
    nk_tols::Vector{Float64}    # NK error bounds, in order
end
SolverInfo() = SolverInfo(false, Float64[], Float64[])

"max |a - b| without allocating."
function maxabsdiff(a::Vector{Float64}, b::Vector{Float64})
    t = 0.0
    @inbounds for i in eachindex(a)
        t = max(t, abs(a[i] - b[i]))
    end
    return t
end

"Tolerance adjusted to the magnitude of v, as in the MATLAB code: tol·10^⌈log10 max|v|⌉."
scaled_tol(tol::Float64, v::Vector{Float64}) = tol * exp10(ceil(log10(maximum(abs, v))))

"""
    solve_sa!(v, pk, Γ; opts, β=nothing, info) -> info

Successive approximations: `v ← Γ(v)` until convergence. If `β` is given,
stop prematurely (after `sa_min` steps) when the observed contraction ratio
`tol_i/tol_{i-1}` is within `tol_ratio` of `β` — the signal to hand over to
Newton–Kantorovich. Mutates `v` and `pk`.
"""
function solve_sa!(v::Vector{Float64}, pk::Vector{Float64}, Γ::BellmanOperator;
                   opts::SolverOptions=SolverOptions(), β::Union{Nothing,Float64}=nothing,
                   info::SolverInfo=SolverInfo())
    v1 = similar(v)
    sa_max = max(opts.sa_min, opts.sa_max)
    tol_prev = NaN
    for i in 1:sa_max
        apply!(v1, pk, Γ, v)
        tol = maxabsdiff(v1, v)
        rtol = tol / (i == 1 ? tol : tol_prev)
        copyto!(v, v1)                          # accept the SA step
        push!(info.sa_tols, tol)
        opts.verbose > 1 && @printf("  SA %4d   tol %16.10e   ratio %10.6f\n", i, tol, rtol)
        if i >= opts.sa_min
            if β !== nothing && abs(β - rtol) < opts.tol_ratio
                opts.verbose > 0 && @printf("SA stopped after %d iterations (contraction ratio ≈ β): switching to NK\n", i)
                return info
            end
            if tol < scaled_tol(opts.sa_tol, v)
                info.converged = true
                opts.verbose > 0 && @printf("SA converged after %d iterations, tol %g\n", i, tol)
                return info
            end
        end
        tol_prev = tol
    end
    opts.verbose > 0 && println("SA reached maximum number of iterations")
    return info
end

"""
    solve_nk!(v, pk, Γ; opts, info) -> info

Newton–Kantorovich iterations on the fixed-point equation `v = Γ(v)`:

    v ← v − (I − Γ′(v))⁻¹ (v − Γ(v))

followed by one extra SA step for stability and an exact error bound
(as in `dpsolver.nk`). Mutates `v` and `pk`.
"""
function solve_nk!(v::Vector{Float64}, pk::Vector{Float64}, Γ::BellmanOperator;
                   opts::SolverOptions=SolverOptions(), info::SolverInfo=SolverInfo())
    n = length(v)
    v1 = similar(v)
    for i in 1:opts.nk_max
        apply!(v1, pk, Γ, v)
        F = lu(sparse(I, n, n) - frechet(Γ, pk))    # I − Γ′, sparse LU
        v .-= F \ (v .- v1)                         # Newton step
        apply!(v1, pk, Γ, v)                        # extra SA step: stability + error bound
        tol = maxabsdiff(v, v1)
        copyto!(v, v1)
        push!(info.nk_tols, tol)
        opts.verbose > 1 && @printf("  NK %4d   tol %16.10e\n", i, tol)
        if tol < scaled_tol(opts.nk_tol, v)
            info.converged = true
            opts.verbose > 0 && @printf("NK converged after %d iterations, tol %g\n", i, tol)
            return info
        end
    end
    opts.verbose > 0 && println("NK reached maximum number of iterations")
    return info
end

"""
    solve_poly!(v, pk, Γ; opts) -> SolverInfo

Poly-algorithm (translation of `dpsolver.poly`): alternate SA and NK stages
up to `max_fxpiter` times until the NK stage converges. `v` holds the initial
guess on entry and the fixed point on exit; `pk` the implied keep-probabilities.
"""
function solve_poly!(v::Vector{Float64}, pk::Vector{Float64}, Γ::BellmanOperator;
                     opts::SolverOptions=SolverOptions())
    info = SolverInfo()
    β = Γ.m.β
    for k in 1:opts.max_fxpiter
        opts.verbose > 0 && @printf("Contraction iterations (stage %d)\n", k)
        solve_sa!(v, pk, Γ; opts=opts, β=β, info=info)
        opts.verbose > 0 && @printf("Newton–Kantorovich iterations (stage %d)\n", k)
        info.converged = false
        solve_nk!(v, pk, Γ; opts=opts, info=info)
        info.converged && return info
    end
    @warn "solve_poly!: no convergence after $(opts.max_fxpiter) SA/NK stages"
    return info
end

"""
    solve(m::Model; space=:ev, v0=zeros(m.n), opts) -> (v, pk, Γ, info)

Convenience wrapper: build the Bellman operator for `m` (`:ev` or `:iv`
space) and solve for its fixed point starting from `v0`.
"""
function solve(m::Model; space::Symbol=:ev, v0::Vector{Float64}=zeros(m.n),
               opts::SolverOptions=SolverOptions())
    Γ = space === :ev ? EVBellman(m) :
        space === :iv ? IVBellman(m) :
        throw(ArgumentError("space must be :ev or :iv"))
    v = copy(v0)
    pk = zeros(m.n)
    info = solve_poly!(v, pk, Γ; opts=opts)
    return v, pk, Γ, info
end
