# ==========================================================================
# run_bbl.jl — BBL moment-inequality criterion surface
#
# Translation of run_bbl.m (demonstration; bbl.estim was a TODO upstream):
# the first-stage policy is the exact solution at the true parameters; the
# perturbed policies shift the replacement probability up/down beyond a
# moving threshold (PolicyPerturbationType = 3 in the MATLAB code). The BBL
# criterion penalizes parameter values at which some perturbed policy beats
# the "estimated" one.
#
# Run:  julia run_bbl.jl       (figures saved to figures/)
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf
ENV["GKSwstype"] = "100"
using Plots

n_pic_grid = 20
npert = 40

m = Model()
_, ccp, _, _ = solve(m; space=:iv)        # first-stage "estimated" policy
pp, _, _ = eqb(m, ccp)                    # stationary distribution = H(x) weights

# perturbed policies: step the keep probability up/down past a moving threshold
pert = Vector{Vector{Float64}}(undef, npert)
for j in 1:npert
    pj = copy(ccp)
    jj = 50 + 2j                          # threshold index
    if iseven(j)
        pj[jj+1:end] .= ccp[jj+1:end] .+ (1 - ccp[jj]) / 2
    else
        pj[jj+1:end] .= ccp[jj+1:end] .- (1 - ccp[jj]) / 2
    end
    pert[j] = clamp.(pj, 1e-10, 1 - 1e-10)
end

mkpath(joinpath(@__DIR__, "figures"))
g = mileage_grid(m)
plt = plot(title="Estimated and perturbed policy functions",
           xlabel="State space", ylabel="Probability of keeping", ylim=(0, 1), legend=false)
for pj in pert
    plot!(plt, g, pj, lw=0.5, color=:black)
end
plot!(plt, g, ccp, lw=2, color=:red)
savefig(plt, joinpath(@__DIR__, "figures", "bbl_policies.png"))

# sanity check: criterion must vanish at the true parameters
check = bbl_objective(m, ccp, pert, pp)
@printf("BBL criterion at true parameters: %1.5e (should be zero)\n", check)

# criterion surface over (c, RC)
cgrid = range(1.0, 5.0, length=n_pic_grid)
RCgrid = range(5.0, 15.0, length=n_pic_grid)
surfvals = zeros(n_pic_grid, n_pic_grid)
for (i, ci) in enumerate(cgrid)
    @printf("%2d ", i)
    for (j, rcj) in enumerate(RCgrid)
        surfvals[j, i] = bbl_objective(update(m; c=ci, RC=rcj), ccp, pert, pp)
        print(".")
    end
    println()
end

plt2 = surface(cgrid, RCgrid, surfvals, xlabel="c parameter", ylabel="RC parameter",
               title="BBL moment-inequality objective")
savefig(plt2, joinpath(@__DIR__, "figures", "bbl_surface.png"))
println("Saved figures/bbl_policies.png and figures/bbl_surface.png")
