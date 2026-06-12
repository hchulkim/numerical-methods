# ==========================================================================
# run_msm.jl — Method of Simulated Moments criterion surface
#
# Translation of run_msm.m (demonstration; msm.estim was a TODO upstream):
# moments are the equilibrium fractions of the fleet in each mileage bin;
# "data" moments are computed from the model solution at the true
# parameters (best case for MSM), and the criterion surface is traced over
# a (c, RC) grid.
#
# Run:  julia run_msm.jl       (figure saved to figures/)
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf
using Random
using LinearAlgebra
using Statistics
ENV["GKSwstype"] = "100"
using Plots

use_solution_for_sims = false    # false: simulate moments;  true: exact stationary distribution
update_weighting_matrix = true
n_pic_grid = 20
nsims, Tsims, scale = 1000, 1000, 100.0

m = Model()

# "data" moments: exact stationary distribution at the true parameters
data_moments, _ = msm_moments(m; use_solution=true, scale=scale)

# weighting matrix
if use_solution_for_sims || !update_weighting_matrix
    W = Matrix(1.0I, m.n - 1, m.n - 1)
    Wlabel = use_solution_for_sims ? "computed moments" : "identity weighting matrix"
else
    _, mom_indiv = msm_moments(m; use_solution=false, nsims=nsims, Tsims=Tsims,
                               scale=scale, rng=Xoshiro(12345))
    dev = mom_indiv .- data_moments'         # per-bus moment conditions
    C = cov(dev)
    if cond(C) > 1e5
        @printf("Condition number %1.3e: switching to diagonal weighting matrix\n", cond(C))
        W = Diagonal(1.0 ./ var(dev, dims=1)[:]) |> Matrix
    else
        W = inv(C)
    end
    Wlabel = "updated weighting matrix"
end

# criterion surface over (c, RC)
cgrid = range(2.0, 3.0, length=n_pic_grid)
RCgrid = range(11.0, 12.0, length=n_pic_grid)
surfvals = zeros(n_pic_grid, n_pic_grid)
for (i, ci) in enumerate(cgrid)
    @printf("%2d ", i)
    for (j, rcj) in enumerate(RCgrid)
        mi = update(m; c=ci, RC=rcj)
        surfvals[j, i] = msm_objective(data_moments, W, mi;
                                       use_solution=use_solution_for_sims,
                                       nsims=nsims, Tsims=Tsims, scale=scale,
                                       rng=Xoshiro(12345))
        print(".")
    end
    println()
end

imin = argmin(surfvals)
@printf("\nGrid minimum at c=%.3f, RC=%.3f (true: c=%.3f, RC=%.3f)\n",
        cgrid[imin[2]], RCgrid[imin[1]], m.c, m.RC)

mkpath(joinpath(@__DIR__, "figures"))
plt = surface(cgrid, RCgrid, surfvals, xlabel="c parameter", ylabel="RC parameter",
              title="MSM objective ($Wlabel)")
savefig(plt, joinpath(@__DIR__, "figures", "msm_surface.png"))
println("Saved figures/msm_surface.png")
