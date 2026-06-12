# ==========================================================================
# run_npl.jl — Estimate the model on Rust's bus data with NPL, compare NFXP
#
# Translation of run_npl.m:
#   1. frequency estimates of the mileage process,
#   2. first-stage CCPs from a flexible logit in mileage,
#   3. NPL iterations (K=1 is the Hotz–Miller two-step CCP estimator),
#   4. NFXP full MLE for comparison,
#   5. solve pk = Ψ(pk) and compare with the Bellman fixed point CCPs.
#
# Run:  julia run_npl.jl       (figure saved to figures/)
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf
ENV["GKSwstype"] = "100"
using Plots

Kmax = 10

m0 = Model(n=175)
data = readbusdata(m0)
N = nobs(data)

# Section 1: first-step frequency estimator of the mileage process
m = update(m0; p=freq_transition(data))

# Section 2: initial CCPs from a flexible logit (pk_init = 2 in run_npl.m)
println("Initialize CCPs with flexible logit (2nd-degree polynomial in mileage)")
pk0 = flexible_logit_ccps(data, m; degree=2)

println("\n", "*"^72)
println("Method: Nested Pseudo Likelihood (NPL)")
@printf("Beta = %g,  n = %d,  sample size = %d\n", m.β, m.n, N)
println("*"^72)
res_npl, pk_npl, K = estim_npl(data, m; θ0=[0.0, 0.0], pk0=pk0, Kmax=Kmax)
println("\nConverged NPL estimates:")
print_estimates(res_npl)

# NFXP for comparison
println("\n", "*"^72)
println("Method: Nested Fixed Point algorithm (NFXP, two-step partial MLE)")
println("*"^72)
res_nfxp = estim_nfxp(data, m; est_p=false)
print_estimates(res_nfxp)

# Section 3: solve the estimated model with Ψ iterations and the poly-algorithm
mhat = res_npl.m
pk_psi, niter = solve_npl(mhat, pk0)
_, pk_fxp, _, _ = solve(mhat; space=:ev)
@printf("\nmax |Ψ-fixed-point − Γ-fixed-point CCPs| = %1.3e (Ψ converged in %d iterations)\n",
        maximum(abs, pk_psi .- pk_fxp), niter)

mkpath(joinpath(@__DIR__, "figures"))
g = mileage_grid(mhat)
plt = plot(g, 1 .- pk0, ls=:dashdot, lw=2, label="Initial CCP",
           title="Replacement probability, K=$K",
           xlabel="Mileage grid", ylabel="Replacement probability", ylim=(0, 0.16))
plot!(plt, g, 1 .- pk_npl, lw=2, color=:black, label="Last evaluation of Psi")
plot!(plt, g, 1 .- pk_psi, lw=2, color=:blue, label="Fixed point of Psi")
plot!(plt, g, 1 .- pk_fxp, lw=2, ls=:dash, color=:red, label="Fixed point of Gamma")
savefig(plt, joinpath(@__DIR__, "figures", "npl_ccps.png"))
println("Saved figures/npl_ccps.png")
