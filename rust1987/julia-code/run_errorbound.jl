# ==========================================================================
# run_errorbound.jl — Convergence of SA vs Newton–Kantorovich
#
# Translation of run_errorbound.m: plots the error bound (and its log)
# against the iteration count for a range of discount factors. With pure
# SA the log-error declines linearly with slope log(β); the poly-algorithm
# switches to NK and converges quadratically in a handful of steps.
#
# Run:  julia run_errorbound.jl       (figures saved to figures/)
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf
ENV["GKSwstype"] = "100"          # headless GR
using Plots

do_nk = false                     # false: pure SA;  true: poly (SA + NK)

mkpath(joinpath(@__DIR__, "figures"))
betavec = [0.95, 0.99, 0.999, 0.9999, 0.99999]

plt1 = plot(xlabel="Iteration count", ylabel="Error bound",
            title="Error bound vs iteration count", legend=:topright)
plt2 = plot(xlabel="Iteration count", ylabel="log(Error bound)",
            title="Log of error bound vs iteration count", legend=:bottomleft)

for β in betavec
    m = Model(n=90, β=β)
    opts = do_nk ? SolverOptions() : SolverOptions(sa_min=250, sa_max=250)
    v, pk, Γ, info = solve(m; space=:iv, opts=opts)

    tols = do_nk ? [info.sa_tols; info.nk_tols] : info.sa_tols
    label = "beta=$β"
    if !do_nk
        # regression slope of log(error) on iteration ≈ log(β)
        y = log.(filter(>(0), tols))
        t = 1:length(y)
        b = [ones(length(y)) t] \ y
        label = @sprintf("slope=%8.5f, log(beta)=%8.5f", b[2], log(β))
    end
    plot!(plt1, 1:length(tols), tols, label="beta=$β", lw=1.5)
    plot!(plt2, 1:length(tols), log.(tols), label=label, lw=1.5)
end

savefig(plt1, joinpath(@__DIR__, "figures", "errorbound.png"))
savefig(plt2, joinpath(@__DIR__, "figures", "errorbound_log.png"))
println("Saved figures/errorbound.png and figures/errorbound_log.png")
