# ==========================================================================
# output.jl — Estimation output tables (translation of output.m)
# ==========================================================================

"""
    print_estimates(res::EstimationResult; truevalues=nothing)

Parameter estimates with standard errors and t-statistics
(format of `output.estimates`).
"""
function print_estimates(res::EstimationResult; truevalues=nothing)
    if truevalues === nothing
        @printf("    %-12s %13s %13s %13s\n", "Param.", "Estimates", "s.e.", "t-stat")
    else
        @printf("    %-12s %13s %13s %13s %13s\n", "Param.", "True Value", "Estimates", "s.e.", "t-stat")
    end
    println("-"^72)
    for (i, nm) in enumerate(res.names)
        if truevalues === nothing
            @printf("    %-12s %13.4f %13.4f %13.4f\n", nm, res.θ[i], res.se[i], res.θ[i] / res.se[i])
        else
            @printf("    %-12s %13.4f %13.4f %13.4f %13.4f\n", nm, truevalues[i], res.θ[i], res.se[i], res.θ[i] / res.se[i])
        end
    end
    println("-"^72)
    @printf("log-likelihood    = %12.4f\n", res.loglike)
    @printf("runtime (seconds) = %12.4f\n", res.runtime)
    @printf("major iterations  = %d,  function evaluations = %d,  converged = %s\n",
            res.iterations, res.f_calls, res.converged)
end

"""
    print_mc_tables(results::Vector{EstimationResult}, mtrue::Model)

Monte Carlo summary tables: parameter bias/MCSD and numerical performance
(format of `output.table_mc` / `output.table_np`).
"""
function print_mc_tables(results::Vector{EstimationResult}, mtrue::Model)
    conv = [r for r in results if r.converged]
    isempty(conv) && (println("No converged Monte Carlo runs!"); return)
    # rare top mileage increments may be absent in small samples, changing
    # the number of estimated transition probabilities — tabulate the modal case
    lens = [length(r.θ) for r in conv]
    klen = lens[argmax([count(==(l), lens) for l in lens])]
    ok = [r for r in conv if length(r.θ) == klen]
    length(ok) < length(conv) &&
        @printf("(%d converged runs estimated a different number of transition probabilities and are not tabulated)\n",
                length(conv) - length(ok))
    names = ok[1].names
    truevals = [mtrue.RC; mtrue.c; mtrue.p][1:length(names)]

    println("\nTable 1: Parameter estimates")
    println("-"^84)
    @printf("%-14s %14s %14s %14s %14s\n", "", "True value", "Estimate", "MCSD", "Bias")
    println("-"^84)
    for (j, nm) in enumerate(names)
        est = [r.θ[j] for r in ok]
        @printf("%-14s %14.3f %14.3f %14.3f %14.4f\n",
                nm, truevals[j], mean(est), std(est), mean(est) - truevals[j])
    end
    @printf("%-14s %14.3f\n", "log-likelihood", mean(r.loglike for r in ok))
    println("-"^84)

    println("\nTable 2: Numerical performance")
    println("-"^84)
    @printf("%20s %14s %14s %14s\n", "Runs converged", "CPU time (s)", "Major iter", "Func. evals")
    println("-"^84)
    @printf("%14d/%-5d %14.4f %14.1f %14.1f\n",
            length(conv), length(results),
            mean(r.runtime for r in conv),
            mean(r.iterations for r in conv),
            mean(r.f_calls for r in conv))
    println("-"^84)
end
