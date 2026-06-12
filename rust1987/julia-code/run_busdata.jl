# ==========================================================================
# run_busdata.jl — Estimate Rust's engine replacement model on his bus data
#
# Translation of run_busdata.m: two-step partial MLE and full MLE with
# NFXP, and the MPEC variant (JuMP + Ipopt) for comparison.
#
# Run:  julia run_busdata.jl
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf

# starting values for (RC, c) — as in run_busdata.m
m0 = Model(RC=0.0, c=0.0, n=175)

data = readbusdata(m0)

println("Structural estimation using bus data from Rust (1987)")
@printf("Beta        = %10.5f\n", m0.β)
@printf("n           = %10d\n", m0.n)
@printf("Sample size = %10d\n\n", nobs(data))

for (label, est_p) in [("NFXP (two-step partial MLE)", false),
                       ("NFXP (full MLE)", true)]
    println("\nMethod: $label")
    res = estim_nfxp(data, m0; est_p=est_p)
    print_estimates(res)
end

println("\nMethod: MPEC (two-step partial MLE, JuMP + Ipopt)")
res_mpec = estim_mpec(data, m0)
print_estimates(res_mpec)
