# ==========================================================================
# nfxp.jl — Nested Fixed Point maximum likelihood estimation
#
# Julia translation of zurcher.ll + nfxp.estim
#
# Outer loop: Newton trust-region on θ = (RC, c[, p₁…p_{J-1}]) with
#             analytic gradient and BHHH Hessian approximation.
# Inner loop: poly-algorithm fixed point of the EV-space Bellman operator,
#             warm-started from the previous evaluation.
#
# Gradient via the implicit function theorem: at the fixed point ev(θ),
#     dev/dθ = (I − Γ′(ev))⁻¹ · ∂Γ/∂θ,
# reusing the sparse LU factorization of I − Γ′ from the Newton solver.
# ==========================================================================

"""
    NFXPCache(n)

Mutable state shared across likelihood evaluations: the fixed point `ev` is
kept and reused as the next starting value (warm start), which lets the
inner solver converge in a handful of Newton steps once the outer optimizer
is near the optimum. Replaces the MATLAB `global ev0`.
"""
mutable struct NFXPCache
    ev::Vector{Float64}
    pk::Vector{Float64}
end
NFXPCache(n::Int) = NFXPCache(zeros(n), zeros(n))

"""
    nfxp_fgh!(Fv, G, H, θ, m0, data, cnt, cache; est_p, opts) -> f or nothing

Negative mean log-likelihood `f`, with analytic gradient `G` and BHHH
Hessian approximation `H` filled in-place when non-`nothing` (the contract
of `Optim.only_fgh!`).

θ = `[RC, c]` (`est_p=false`, transition probabilities held at `m0.p`)
or  `[RC, c, p₁, …, p_{J-1}]` (`est_p=true`, full MLE).
"""
function nfxp_fgh!(Fv, G, H, θ::Vector{Float64}, m0::Model, data::BusData,
                   cnt::ChoiceCounts, cache::NFXPCache;
                   est_p::Bool, opts::SolverOptions=SolverOptions())
    p = est_p ? abs.(θ[3:end]) : m0.p     # |p|: same positivity hack as MATLAB
    m = update(m0; RC=θ[1], c=θ[2], p=p)
    n = m.n; β = m.β; N = cnt.N

    # ---- inner loop: solve the fixed point (warm start from cache) --------
    Γ = EVBellman(m)
    solve_poly!(cache.ev, cache.pk, Γ; opts=opts)
    ev, pk, V, pf = cache.ev, cache.pk, Γ.V, Γ.pf
    J = length(pf)

    # ---- f: negative mean log-likelihood, aggregated by state: O(n) -------
    ll = 0.0
    @inbounds for x in 1:n
        cnt.keep[x] > 0 && (ll += cnt.keep[x] * log(pk[x]))
        cnt.repl[x] > 0 && (ll += cnt.repl[x] * log1p(-pk[x]))
    end
    if est_p
        @inbounds for j in 1:J
            cnt.dx[j] > 0 && (ll += cnt.dx[j] * log(pf[j]))
        end
    end
    f = -ll / N

    # ---- gradient and BHHH Hessian -----------------------------------------
    if G !== nothing || H !== nothing
        np = est_p ? J - 1 : 0
        K = 2 + np

        # STEP 1: ∂Γ/∂θ holding ev fixed (n × K).
        # Utility parameters enter through V(y) = logsumexp(vK(y), vR):
        #   ∂V(y)/∂RC = −(1−pk(y)),   ∂V(y)/∂c = −0.001·(y−1)·pk(y),
        # then premultiply by P_keep (Γ = P_keep∘logsumexp in EV space).
        dΓdθ = zeros(n, K)
        tmp = Vector{Float64}(undef, n)
        @inbounds for y in 1:n
            tmp[y] = -(1.0 - pk[y])
        end
        keep_mul!(view(dΓdθ, :, 1), pf, tmp)
        @inbounds for y in 1:n
            tmp[y] = -0.001 * (y - 1) * pk[y]
        end
        keep_mul!(view(dΓdθ, :, 2), pf, tmp)
        # Free transition probabilities: with π_J = 1 − Σ_j π_j,
        #   ∂Γ(x)/∂p_j = V(min(x+j−1, n)) − V(min(x+J−1, n)),
        # exact including the clamped boundary rows.
        if est_p
            @inbounds for j in 1:np, x in 1:n
                dΓdθ[x, 2+j] = V[min(x + j - 1, n)] - V[min(x + J - 1, n)]
            end
        end

        # STEP 2: implicit function theorem (one sparse LU, K right-hand sides)
        Flu = lu(sparse(I, n, n) - frechet(Γ, pk))
        dev = Flu \ dΓdθ                                       # n × K

        # STEP 3: scores. With vK(x) = u_keep(x) + β·ev(x), vR = u_repl + β·ev(1):
        #   Δdv(x,·) = ∂(vK(x) − vR)/∂θ = du_keep − du_repl + β·(dev(x,·) − dev(1,·))
        #   score_i  = (1{keep_i} − pk(x_i)) · Δdv(x_i,·)  (+ transition part below)
        Δdv = Matrix{Float64}(undef, n, K)
        @inbounds for x in 1:n
            Δdv[x, 1] = 1.0 + β * (dev[x, 1] - dev[1, 1])      # du_keep − du_repl = 0 − (−1)
            Δdv[x, 2] = -0.001 * (x - 1) + β * (dev[x, 2] - dev[1, 2])
            for j in 1:np
                Δdv[x, 2+j] = β * (dev[x, 2+j] - dev[1, 2+j])
            end
        end
        score = Matrix{Float64}(undef, N, K)
        @inbounds for i in 1:N
            x = data.x[i]
            w = data.d[i] ? -pk[x] : 1.0 - pk[x]               # 1{keep} − pk(x)
            for k in 1:K
                score[i, k] = w * Δdv[x, k]
            end
        end
        if est_p
            # ∂ log π(dx1) / ∂p_j = 1{dx1 = j−1}/π_j − 1{dx1 = J−1}/π_J
            @inbounds for i in 1:N
                dx = data.dx1[i]
                if dx < J - 1
                    score[i, 2+dx+1] += 1.0 / pf[dx+1]
                else
                    for j in 1:np
                        score[i, 2+j] -= 1.0 / pf[J]
                    end
                end
            end
        end
        if G !== nothing
            @inbounds for k in 1:K
                s = 0.0
                for i in 1:N
                    s += score[i, k]
                end
                G[k] = -s / N
            end
        end
        if H !== nothing
            mul!(H, score', score)
            H ./= N
        end
    end
    return Fv === nothing ? nothing : f
end

"""
Practical convergence check: BHHH is only an approximation to the Hessian,
so on small samples the trust region can stall with the gradient at ~1e-6
— converged for all practical purposes (MATLAB's fminunc with TolFun=1e-5
reports success there too). Accept either Optim's own criteria or a small
final gradient.
"""
practically_converged(r) = Optim.converged(r) || Optim.g_residual(r) < 1e-4

"Estimation results returned by `estim_nfxp` / `estim_npl`."
struct EstimationResult
    names::Vector{String}
    θ::Vector{Float64}
    se::Vector{Float64}
    Avar::Matrix{Float64}
    loglike::Float64        # log-likelihood at the optimum (not the mean)
    runtime::Float64        # seconds
    iterations::Int
    f_calls::Int
    converged::Bool
    m::Model                # model at the estimates
end

"""
    estim_nfxp(data, m0; est_p=true, solver_opts, optim_opts) -> EstimationResult

NFXP maximum likelihood (translation of `nfxp.estim`):

1. first-step frequency estimates of the mileage process `p`,
2. partial MLE of the utility parameters (RC, c) holding `p` fixed,
3. if `est_p`, full MLE of (RC, c, p) starting from steps 1–2.

Standard errors from the inverse BHHH information at the estimates.
"""
function estim_nfxp(data::BusData, m0::Model; est_p::Bool=true,
                    solver_opts::SolverOptions=SolverOptions(),
                    optim_opts=Optim.Options(g_tol=1e-7, x_abstol=1e-8))
    t0 = time()
    N = nobs(data)

    # STEP 1: frequency estimator of the mileage process
    p̂ = freq_transition(data)
    m = update(m0; p=p̂)
    cnt = ChoiceCounts(data, m.n, length(p̂) + 1)
    cache = NFXPCache(m.n)

    # STEP 2a: partial MLE — utility parameters only
    obj1 = Optim.only_fgh!((F, G, H, θ) ->
        nfxp_fgh!(F, G, H, θ, m, data, cnt, cache; est_p=false, opts=solver_opts))
    r1 = optimize(obj1, [m0.RC, m0.c], NewtonTrustRegion(), optim_opts)
    θ̂ = Optim.minimizer(r1)
    m = update(m; RC=θ̂[1], c=θ̂[2])
    iterations = Optim.iterations(r1)
    f_calls = Optim.f_calls(r1)
    converged = practically_converged(r1)
    rfinal = r1

    # STEP 2b: full MLE — utility and transition parameters jointly
    if est_p
        obj2 = Optim.only_fgh!((F, G, H, θ) ->
            nfxp_fgh!(F, G, H, θ, m, data, cnt, cache; est_p=true, opts=solver_opts))
        r2 = optimize(obj2, [θ̂; p̂], NewtonTrustRegion(), optim_opts)
        θ̂ = Optim.minimizer(r2)
        m = update(m; RC=θ̂[1], c=θ̂[2], p=abs.(θ̂[3:end]))
        iterations += Optim.iterations(r2)
        f_calls += Optim.f_calls(r2)
        converged = converged && practically_converged(r2)
        rfinal = r2
    end

    # variance-covariance: inverse BHHH information, Avar = (N·h)⁻¹
    K = length(θ̂)
    G = zeros(K); H = zeros(K, K)
    f = nfxp_fgh!(0.0, G, H, θ̂, m, data, cnt, cache; est_p=est_p, opts=solver_opts)
    Avar = inv(H * N)

    names = est_p ? ["RC"; "c"; ["p$(j)" for j in 1:K-2]] : ["RC", "c"]
    return EstimationResult(names, θ̂, sqrt.(diag(Avar)), Avar, -f * N,
                            time() - t0, iterations, f_calls, converged, m)
end
