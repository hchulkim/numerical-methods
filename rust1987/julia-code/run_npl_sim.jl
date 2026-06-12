# ==========================================================================
# run_npl_sim.jl — NPL vs NFXP on a large simulated panel
#
# Translation of run_npl_sim.m: solve the model at Rust's Table X
# parameters, simulate N=5000 buses for T=119 months, then estimate with
# NPL (flexible-logit starting CCPs) and NFXP.
#
# Run:  julia run_npl_sim.jl
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf
using Random

Kmax = 10
N, T = 5000, 119

# Section 0: solve the model and simulate data
mtrue = Model(n=175)
_, pk_true, _, _ = solve(mtrue; space=:ev)
data = BusData(simdata(N, T, mtrue, pk_true; rng=Xoshiro(123)))

# Section 1: frequency estimator of the mileage process
m = update(mtrue; p=freq_transition(data))

# Initial CCPs: flexible logit, 4th-degree polynomial (pk_init = 2)
println("Initialize CCPs with flexible logit (4th-degree polynomial in mileage)")
pk0 = flexible_logit_ccps(data, m; degree=4)

println("\n", "*"^72)
println("Method: Nested Pseudo Likelihood (NPL)")
@printf("Beta = %g,  n = %d,  sample size = %d\n", m.β, m.n, nobs(data))
println("*"^72)
res_npl, pk_npl, K = estim_npl(data, m; θ0=[0.0, 0.0], pk0=pk0, Kmax=Kmax)
println("\nConverged NPL estimates (true: RC=$(mtrue.RC), c=$(mtrue.c)):")
print_estimates(res_npl; truevalues=[mtrue.RC, mtrue.c])

println("\n", "*"^72)
println("Method: Nested Fixed Point algorithm (NFXP, full MLE)")
println("*"^72)
res_nfxp = estim_nfxp(data, m; est_p=true)
print_estimates(res_nfxp; truevalues=[mtrue.RC; mtrue.c; pfull(mtrue)[1:length(res_nfxp.θ)-2]])
