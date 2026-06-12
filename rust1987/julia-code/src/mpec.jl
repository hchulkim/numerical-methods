# ==========================================================================
# mpec.jl — Mathematical Programming with Equilibrium Constraints
#
# Julia translation of matlab-code/mpec.m (Su & Judd 2012, Ecta 80(5)),
# using JuMP + Ipopt instead of fmincon.
#
# Instead of nesting a fixed-point solve inside the likelihood (NFXP), MPEC
# treats the expected value function as 175 extra decision variables and
# imposes the Bellman equation as nonlinear equality constraints:
#
#     max_{RC, c, ev}  (1/N) Σ_i log P(d_i | x_i; RC, c, ev)
#     s.t.             ev = Γ(ev; RC, c)            (n equality constraints)
#                      RC ≥ 0, c ≥ 0, −5000 ≤ ev ≤ 0
#
# The constraint Jacobian is sparse (banded, like the transition matrix);
# the MATLAB code supplies sparsity patterns and analytic derivatives to
# fmincon by hand — JuMP's automatic differentiation derives both for us.
#
# As upstream, only the two-step partial MLE is implemented (transition
# probabilities from the first-step frequency estimator).
# ==========================================================================

"""
    estim_mpec(data, m0; silent=true) -> EstimationResult

MPEC estimation of (RC, c) with the mileage process fixed at its first-step
frequency estimates (translation of `mpec.estim`).

Numerical safety: the Bellman constraint needs `log(exp(vK) + exp(vR))`
with values around −2000, where naive `exp` underflows. We anchor on the
keep value: `logsumexp(vK, vR) = vK + log1p(exp(vR − vK))` — exact, smooth,
and the exponent `vR − vK` is a bounded value *difference*, so Ipopt's AD
never sees an overflowing intermediate. (The MATLAB/AMPL formulations rely
on analytic derivatives resp. luck for the same reason.)

Standard errors are computed afterwards from the BHHH information of the
unconstrained likelihood at the MPEC estimates (the MATLAB version returns
`NaN` here).
"""
function estim_mpec(data::BusData, m0::Model; silent::Bool=true,
                    ipopt_options::AbstractDict=Dict{String,Any}())
    t0 = time()
    N = nobs(data)

    # STEP 1: frequency estimator of the mileage process (as in NFXP)
    p̂ = freq_transition(data)
    m = update(m0; p=p̂)
    pf = pfull(m)
    n, J, β = m.n, length(pf), m.β
    cnt = ChoiceCounts(data, n, J)

    # STEP 2: constrained likelihood maximization
    jm = JuMP.Model(Ipopt.Optimizer)
    silent && JuMP.set_silent(jm)
    JuMP.set_attribute(jm, "tol", 1e-8)
    # β = 0.9999 makes I − βΓ′ nearly singular (condition ~ 1/(1−β) = 10⁴),
    # and the exact Hessian of the Lagrangian is indefinite along the
    # resulting ridge — Ipopt's inertia correction then cripples the Newton
    # steps and the barrier crawls (thousands of iterations, stalls early).
    # The positive-definite L-BFGS approximation tracks the ridge instead:
    # converges in ~30 iterations to the NFXP optimum.
    JuMP.set_attribute(jm, "hessian_approximation", "limited-memory")
    # don't let the "acceptable level" heuristic stop a slow run early
    JuMP.set_attribute(jm, "acceptable_iter", 0)
    JuMP.set_attribute(jm, "max_iter", 3000)
    for (k, v) in ipopt_options
        JuMP.set_attribute(jm, k, v)
    end

    # feasible warm start: solve the fixed point once at the starting θ, so
    # the Bellman constraints hold exactly at the initial point (one cheap
    # poly-algorithm solve; helps Ipopt enormously on the β≈1 ridge)
    ev0 = zeros(n); pk0 = zeros(n)
    solve_poly!(ev0, pk0, EVBellman(m))

    # θ ≥ 0 as in mpec.estim; the ev box is cautionary and must not bind
    # (MATLAB uses [-5000, 0], but at θ→0 the fixed point is positive:
    #  log(2)/(1−β) ≈ +6931, so a wider box keeps the warm start feasible)
    JuMP.@variable(jm, RC ≥ 0, start = m0.RC)
    JuMP.@variable(jm, c ≥ 0, start = m0.c)
    JuMP.@variable(jm, -1e5 ≤ ev[1:n] ≤ 1e5)
    for x in 1:n
        JuMP.set_start_value(ev[x], ev0[x])
    end

    # choice-specific values and their difference (affine in RC, c, ev)
    JuMP.@expression(jm, vK[x in 1:n], -0.001 * c * (x - 1) + β * ev[x])
    JuMP.@expression(jm, vR, -RC + β * ev[1])
    JuMP.@expression(jm, D[x in 1:n], vR - vK[x])     # vR − vK: bounded, safe to exp

    # Bellman equation in EV space as equality constraints:
    #   ev(x) = Σ_j π_j · logsumexp(vK(y), vR),  y = min(x+j−1, n)
    JuMP.@constraint(jm, bellman[x in 1:n],
        ev[x] == sum(pf[j] * (vK[min(x + j - 1, n)] + log1p(exp(D[min(x + j - 1, n)])))
                     for j in 1:J))

    # mean log-likelihood, aggregated by state (cf. ChoiceCounts):
    #   log P(keep|x) = −log1p(exp(D));  log P(replace|x) = D − log1p(exp(D))
    JuMP.@objective(jm, Max,
        sum(-(cnt.keep[x] + cnt.repl[x]) * log1p(exp(D[x])) + cnt.repl[x] * D[x]
            for x in 1:n if cnt.keep[x] + cnt.repl[x] > 0) / N)

    JuMP.optimize!(jm)

    status = JuMP.termination_status(jm)
    converged = status in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
    θ̂ = [JuMP.value(RC), JuMP.value(c)]
    loglike = JuMP.objective_value(jm) * N

    # s.e. from the BHHH information of the (unconstrained) likelihood at θ̂
    G = zeros(2); H = zeros(2, 2)
    cache = NFXPCache(n)
    nfxp_fgh!(0.0, G, H, θ̂, m, data, cnt, cache; est_p=false)
    Avar = inv(H * N)

    iters = Int(JuMP.barrier_iterations(jm))
    return EstimationResult(["RC", "c"], θ̂, sqrt.(diag(Avar)), Avar, loglike,
                            time() - t0, iters, iters, converged, update(m; RC=θ̂[1], c=θ̂[2]))
end
