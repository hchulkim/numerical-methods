# ==========================================================================
# npl.jl — Nested Pseudo-Likelihood (Aguirregabiria & Mira 2002) and the
#          Hotz–Miller style policy iteration operator Ψ
#
# Julia translation of matlab-code/npl.m
#
# Ψ maps choice probabilities to choice probabilities:
#   1. policy evaluation φ: given CCPs pk₀, the value of behaving according
#      to pk₀ forever solves the *linear* system
#          (I − β·F(pk₀)) Vσ = pv(pk₀),
#      where F is the unconditional transition matrix under pk₀ and pv the
#      expected flow payoff (with the log-correction from T1EV shocks),
#   2. policy improvement Λ: best-respond to Vσ with a logit.
# NPL alternates pseudo-likelihood maximization of θ with one Ψ update.
# ==========================================================================

const EULER = 0.5772156649015328606065120900824024310421    # Euler–Mascheroni γ

"""
    policy_matrix(m, pk) -> SparseMatrixCSC

Unconditional mileage transition under the policy `pk`:
`F = P_keep .* pk + P_replace .* (1-pk)` (row-wise mixture).
"""
function policy_matrix(m::Model, pk::Vector{Float64})
    pf = pfull(m); n = m.n; J = length(pf)
    rows = Vector{Int}(undef, 2n * J); cols = Vector{Int}(undef, 2n * J)
    vals = Vector{Float64}(undef, 2n * J)
    k = 0
    @inbounds for x in 1:n, j in 1:J
        k += 1; rows[k] = x; cols[k] = min(x + j - 1, n); vals[k] = pf[j] * pk[x]
        k += 1; rows[k] = x; cols[k] = min(j, n);         vals[k] = pf[j] * (1.0 - pk[x])
    end
    return sparse(rows, cols, vals, n, n)
end

"Sparse LU of `(I − β·F(pk₀))` — the policy-evaluation operator, reused across θ."
policy_lu(m::Model, pk0::Vector{Float64}) =
    lu(sparse(I, m.n, m.n) - m.β * policy_matrix(m, pk0))

"""
    phi(m, pk0, Flu) -> Vσ

Policy evaluation (`npl.phi`): value of following the CCPs `pk0` forever,
including the conditional expectation `γ − log pk` of the T1EV taste shocks.
`Flu` is the factorization from `policy_lu(m, pk0)`.
"""
function phi(m::Model, pk0::Vector{Float64}, Flu)
    pv = Vector{Float64}(undef, m.n)
    @inbounds for x in 1:m.n
        vK = u_keep(m, x) + EULER - log(pk0[x])
        vR = u_replace(m) + EULER - log1p(-pk0[x])
        pv[x] = pk0[x] * vK + (1.0 - pk0[x]) * vR
    end
    return Flu \ pv
end

"""
    lambda!(pk, m, Vσ, buf) -> pk

Policy improvement (`npl.lambda`): logit best response to the continuation
values `Vσ`. `buf` is an n-buffer for `P_keep*Vσ`.
"""
function lambda!(pk::Vector{Float64}, m::Model, Vσ::Vector{Float64}, buf::Vector{Float64})
    pf = pfull(m); β = m.β
    keep_mul!(buf, pf, Vσ)
    vR = u_replace(m) + β * replace_expectation(pf, Vσ)
    @inbounds for x in 1:m.n
        vK = u_keep(m, x) + β * buf[x]
        pk[x] = 1.0 / (1.0 + exp(vR - vK))
    end
    return pk
end

"Policy iteration operator `Ψ = Λ ∘ φ` (`npl.Psi`)."
function Psi(m::Model, pk0::Vector{Float64})
    Vσ = phi(m, pk0, policy_lu(m, pk0))
    pk = similar(pk0)
    return lambda!(pk, m, Vσ, similar(pk0))
end

"""
    solve_npl(m, pk0; tol=1e-12, maxiter=100) -> (pk, niter)

Solve `pk = Ψ(pk)` by successive applications of Ψ (`npl.solve`).
"""
function solve_npl(m::Model, pk0::Vector{Float64}; tol::Float64=1e-12, maxiter::Int=100)
    pk0 = copy(pk0)
    for i in 1:maxiter
        pk1 = Psi(m, pk0)
        t = maxabsdiff(pk1, pk0)
        pk0 = pk1
        t < tol && return pk0, i
    end
    return pk0, maxiter
end

"""
    npl_fgh!(Fv, G, H, θ, m0, data, cnt, pk0, Flu, pk_out, buf) -> f

Pseudo log-likelihood for fixed first-stage CCPs `pk0` (with `Flu` the
corresponding policy-evaluation factorization). θ = [RC, c]. Fills the
updated CCPs Λ(φ(pk0; θ)) into `pk_out` — the Ψ update reused by `estim_npl`.

Analytic gradient (FD-verified): with mΔ(x) = vR − vK(x),
    score_i = (1{keep_i} − pk(x_i)) · ∂mΔ(x_i)/∂θ,    G = mean(score),
and BHHH `H = score′score/N` (cf. `npl.ll`).
"""
function npl_fgh!(Fv, G, H, θ::Vector{Float64}, m0::Model, data::BusData,
                  cnt::ChoiceCounts, pk0::Vector{Float64}, Flu,
                  pk_out::Vector{Float64}, buf::Vector{Float64})
    m = update(m0; RC=θ[1], c=θ[2])
    n = m.n; β = m.β; N = cnt.N; pf = pfull(m)

    Vσ = phi(m, pk0, Flu)
    lambda!(pk_out, m, Vσ, buf)

    ll = 0.0
    @inbounds for x in 1:n
        cnt.keep[x] > 0 && (ll += cnt.keep[x] * log(pk_out[x]))
        cnt.repl[x] > 0 && (ll += cnt.repl[x] * log1p(-pk_out[x]))
    end
    f = -ll / N

    if G !== nothing || H !== nothing
        # ∂Vσ/∂θ = (I − βF)⁻¹ ∂pv/∂θ, holding pk0 (and hence F) fixed:
        #   ∂pv/∂RC = −(1−pk0),   ∂pv/∂c = −0.001·grid .* pk0
        dpvRC = Vector{Float64}(undef, n)
        dpvc = Vector{Float64}(undef, n)
        @inbounds for x in 1:n
            dpvRC[x] = -(1.0 - pk0[x])
            dpvc[x] = -0.001 * (x - 1) * pk0[x]
        end
        wRC = Flu \ dpvRC
        wc = Flu \ dpvc

        # ∂(vR − vK)(x)/∂θ, with vK = u_keep + β·P_keep Vσ, vR = u_repl + β·P_repl Vσ
        dmRC = Vector{Float64}(undef, n)
        dmc = Vector{Float64}(undef, n)
        keep_mul!(buf, pf, wRC)
        rexp = replace_expectation(pf, wRC)
        @inbounds for x in 1:n
            dmRC[x] = -1.0 + β * (rexp - buf[x])
        end
        keep_mul!(buf, pf, wc)
        rexp = replace_expectation(pf, wc)
        @inbounds for x in 1:n
            dmc[x] = 0.001 * (x - 1) + β * (rexp - buf[x])
        end

        score = Matrix{Float64}(undef, N, 2)
        @inbounds for i in 1:N
            x = data.x[i]
            res = (data.d[i] ? -pk_out[x] : 1.0 - pk_out[x])   # 1{keep} − pk
            score[i, 1] = res * dmRC[x]
            score[i, 2] = res * dmc[x]
        end
        if G !== nothing
            G[1] = sum(view(score, :, 1)) / N
            G[2] = sum(view(score, :, 2)) / N
        end
        if H !== nothing
            mul!(H, score', score)
            H ./= N
        end
    end
    return Fv === nothing ? nothing : f
end

"""
    estim_npl(data, m0; θ0=[0.0,0.0], pk0, Kmax=100, verbose=true) -> EstimationResult, pk, K

NPL algorithm (translation of `npl.estim`). Starting from first-stage CCPs
`pk0`, iterate:

1. maximize the pseudo-likelihood over θ = (RC, c), holding pk0 fixed,
2. update the CCPs with one Ψ step at the new θ,

until `max|θ_K − θ_{K-1}| < 1e-6` or `Kmax` iterations. K = 1 is the
Hotz–Miller two-step CCP estimator; K → ∞ converges to the NPL fixed point.
"""
function estim_npl(data::BusData, m0::Model;
                   θ0::Vector{Float64}=[0.0, 0.0],
                   pk0::Vector{Float64}, Kmax::Int=100, verbose::Bool=true)
    t0 = time()
    N = nobs(data)
    cnt = ChoiceCounts(data, m0.n, length(pfull(m0)))
    pk0 = copy(pk0)
    pk = similar(pk0)
    buf = similar(pk0)
    θ = copy(θ0)
    m = m0
    K = 0
    f_calls = 0
    converged = false
    f̂ = NaN

    if verbose
        @printf("%-6s %14s %14s %14s\n", "K", "RC", "c", "log-like")
        println("-"^52)
    end
    local rK
    for outer K in 1:Kmax
        Flu = policy_lu(m0, pk0)
        obj = Optim.only_fgh!((F, G, H, t) ->
            npl_fgh!(F, G, H, t, m0, data, cnt, pk0, Flu, pk, buf))
        rK = optimize(obj, θ, NewtonTrustRegion(), Optim.Options(g_tol=1e-6))
        θnew = Optim.minimizer(rK)
        f̂ = Optim.minimum(rK)
        f_calls += Optim.f_calls(rK)
        npl_metric = maximum(abs, θnew .- θ)

        # Ψ update: pk currently holds Λ(φ(pk0; θ̂)) from the last obj call at θ̂
        nfinal = npl_fgh!(0.0, nothing, nothing, θnew, m0, data, cnt, pk0, Flu, pk, buf)
        copyto!(pk0, pk)
        θ = θnew

        verbose && @printf("%-6d %14.4f %14.4f %14.1f\n", K, θ[1], θ[2], -f̂ * N)
        if npl_metric < 1e-6
            converged = true
            break
        end
    end
    m = update(m0; RC=θ[1], c=θ[2])

    # standard errors from BHHH at the final stage
    G = zeros(2); H = zeros(2, 2)
    Flu = policy_lu(m0, pk0)
    npl_fgh!(0.0, G, H, θ, m0, data, cnt, pk0, Flu, pk, buf)
    Avar = inv(H * N)

    res = EstimationResult(["RC", "c"], θ, sqrt.(diag(Avar)), Avar, -f̂ * N,
                           time() - t0, K, f_calls, converged, m)
    return res, pk0, K
end

"""
    fit_logit(y, X) -> θ

Newton–Raphson logit MLE (replaces `npl.ll_logit` + fminunc); used to build
flexible-logit starting CCPs for NPL.
"""
function fit_logit(y::AbstractVector{<:Real}, X::AbstractMatrix{Float64}; maxiter::Int=100)
    θ = zeros(size(X, 2))
    for _ in 1:maxiter
        η = X * θ
        px = @. 1.0 / (1.0 + exp(-η))
        g = X' * (px .- y)
        W = @. px * (1.0 - px)
        Hm = X' * (W .* X)
        δ = Hm \ g
        θ -= δ
        maximum(abs, δ) < 1e-10 && break
    end
    return θ
end

"""
    flexible_logit_ccps(data, m; degree=2) -> pk0

First-stage CCPs from a logit with a polynomial in (mileage/n), evaluated on
the grid (the `pk_init=2` branch of run_npl.m).
"""
function flexible_logit_ccps(data::BusData, m::Model; degree::Int=2)
    N = nobs(data)
    X = ones(N, degree + 1)
    Xg = ones(m.n, degree + 1)
    for k in 1:degree
        X[:, k+1] = (data.x ./ m.n) .^ k
        Xg[:, k+1] = ((0:m.n-1) ./ m.n) .^ k
    end
    θ = fit_logit(Float64.(data.d), X)            # P(replace | x) logit
    return 1.0 ./ (1.0 .+ exp.(Xg * θ))           # P(keep | x) on the grid
end

# --------------------------------------------------------------------------
# BBL moment-inequality objective (translation of bbl.objective; bbl.estim
# was a TODO stub upstream and is not ported)
# --------------------------------------------------------------------------

"""
    bbl_objective(m, pol0, pol1s, Hw) -> Float64

Squared one-sided deviations of the BBL (2007) moment inequalities: the
first-stage policy `pol0` should yield (weakly) higher values than any
perturbed policy in `pol1s`; violations are penalized under the state
distribution `Hw`.
"""
function bbl_objective(m::Model, pol0::Vector{Float64},
                       pol1s::Vector{Vector{Float64}}, Hw::Vector{Float64})
    V0 = phi(m, pol0, policy_lu(m, pol0))
    res = 0.0
    for pol1 in pol1s
        V1 = phi(m, pol1, policy_lu(m, pol1))
        g = V0 .- V1
        res += dot(Hw, min.(g, 0.0) .^ 2) / length(pol1s)
    end
    return res
end
