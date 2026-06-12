# ==========================================================================
# run_fxp.jl — Solve the model once and print solver iteration detail
#
# Translation of run_fxp.m. Switch `algorithm` between :sa (successive
# approximations only) and :poly (SA + Newton–Kantorovich).
#
# Run:  julia run_fxp.jl
# ==========================================================================

include(joinpath(@__DIR__, "src", "Zurcher.jl"))
using .Zurcher
using Printf

algorithm = :poly          # :sa or :poly

m = Model(n=90)            # Rust's original grid size
opts = SolverOptions(verbose=2, sa_max=1000)

Γ = EVBellman(m)
ev = zeros(m.n)
pk = zeros(m.n)

if algorithm === :sa
    info = solve_sa!(ev, pk, Γ; opts=opts)
elseif algorithm === :poly
    info = solve_poly!(ev, pk, Γ; opts=opts)
else
    error("algorithm must be :sa or :poly")
end

@printf("\nconverged: %s,  SA iterations: %d,  NK iterations: %d\n",
        info.converged, length(info.sa_tols), length(info.nk_tols))
@printf("ev(1) = %.6f,  P(keep | x=1) = %.6f,  P(keep | x=n) = %.6f\n",
        ev[1], pk[1], pk[end])
