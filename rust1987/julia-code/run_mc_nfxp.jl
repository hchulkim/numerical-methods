# ==========================================================================
# run_mc_nfxp.jl — Monte Carlo performance of the NFXP estimator
#
# Translation of run_mc_nfxp.m: simulate nMC samples of N buses over T
# months from the model at Rust's Table X parameters, re-estimate by full
# MLE on each sample, and tabulate bias / Monte Carlo standard deviations
# and numerical performance.
#
# Run:  julia run_mc_nfxp.jl
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf
using Random

nMC = 100                    # number of Monte Carlo samples
N = 50                       # buses per sample
T = 119                      # months per bus

m = Model()                  # DGP: Rust's Table X parameters
_, pk0, _, _ = solve(m; space=:ev)

@printf("Begin Monte Carlo with %d replications (N=%d, T=%d)\n", nMC, N, T)
rng = Xoshiro(301)
results = EstimationResult[]
for i_mc in 1:nMC
    t = @elapsed sim = simdata(N, T, m, pk0; rng=rng)
    data = BusData(sim)
    @printf("i_mc=%3d  time to simulate: %1.5gs\n", i_mc, t)
    push!(results, estim_nfxp(data, m; est_p=true))
end

print_mc_tables(results, m)
