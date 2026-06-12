# ==========================================================================
# run_dem.jl — Equilibrium mileage distribution and engine demand curve
#
# Translation of run_dem.m: computes the equilibrium distribution of
# mileage and the expected annual demand for bus engines as a function of
# the replacement cost RC, for the static (β=0) and dynamic (β=0.9999)
# models. (Known issue inherited from the MATLAB code: the demand curve
# does not exactly replicate Figure 7 of Rust 1987.)
#
# Run:  julia run_dem.jl       (figures saved to figures/)
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf
ENV["GKSwstype"] = "100"
using Plots

mkpath(joinpath(@__DIR__, "figures"))

plt_dem = plot(xlabel="Replacement cost, RC", ylabel="Expected annual engine replacement",
               title="Expected Replacement Demand Function",
               xlim=(0, 12000), ylim=(0, 0.4))

for dynamic in (false, true)
    # estimated parameters at n = 90 (model 11 of Rust's Table IX for the
    # dynamic case; static logit estimates otherwise) — as in run_dem.m
    m = dynamic ?
        Model(n=90, RC=10.0750, c=2.2930, β=0.9999, p=[0.3919, 0.5953]) :
        Model(n=90, RC=7.6358, c=71.5133, β=0.0, p=[0.3919, 0.5953])

    ev, pk, _, _ = solve(m; space=:ev)
    pp, pp_K, pp_R = eqb(m, pk)
    x = range(0, m.maxmiles, length=m.n)

    if dynamic
        @printf("Fraction of bus engines replaced each month : %1.5f\n", sum(pp_R))
        @printf("Mean lifetime of bus engine (months)        : %1.5f\n", 1 / sum(pp_R))
        @printf("Mean mileage at overhaul                    : %1.5f\n", sum(x .* pp_R) / sum(pp_R))
        @printf("Mean mileage since last replacement         : %1.5f\n", sum(x .* pp_K) / sum(pp_K))

        plt_eq = plot(x, [pp_K ./ sum(pp_K) pp_R ./ sum(pp_R)],
                      label=["Pr(x | Keep)" "Pr(x | Replace)"], lw=3,
                      title="Equilibrium distribution: bus mileage",
                      xlabel="Mileage (1000s)", ylabel="Density",
                      xlim=(0, 440), ylim=(0, 0.026))
        savefig(plt_eq, joinpath(@__DIR__, "figures", "equilibrium_distribution.png"))
    end

    # demand for engines as a function of RC (warm-started in RC)
    RCgrid = 1.0:0.5:30.0
    demand = equilibrium_demand(m, RCgrid; v0=copy(ev))
    plot!(plt_dem, RCgrid .* (4343 / m.RC), demand,
          label=dynamic ? "beta=0.9999" : "beta=0", lw=3)
end

savefig(plt_dem, joinpath(@__DIR__, "figures", "demand_curve.png"))
println("Saved figures/equilibrium_distribution.png and figures/demand_curve.png")
