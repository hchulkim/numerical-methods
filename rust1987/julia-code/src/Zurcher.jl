# ==========================================================================
# Zurcher.jl — Rust's (1987) bus engine replacement model in Julia
#
# Translation of the MATLAB code by Iskhakov, Schjerning & Rust
# (matlab-code/, August 2021 version). See README.md for the full map
# between the MATLAB files and this module.
#
# Usage from a script:
#     include("src/Zurcher.jl")
#     using .Zurcher
# ==========================================================================
module Zurcher

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf
using CSV
using DataFrames
using Optim
import JuMP                # qualified: JuMP.Model would clash with our Model
import Ipopt

export Model, update, pfull, mileage_grid, cost, u_keep, u_replace,
    keep_mul!, replace_expectation, statetransition,
    EVBellman, IVBellman, apply!, frechet, iv_to_ev,
    SolverOptions, SolverInfo, solve_sa!, solve_nk!, solve_poly!, solve,
    BusData, readbusdata, freq_transition, ChoiceCounts, nobs,
    NFXPCache, nfxp_fgh!, estim_nfxp, EstimationResult, estim_mpec,
    policy_matrix, policy_lu, phi, lambda!, Psi, solve_npl, npl_fgh!,
    estim_npl, fit_logit, flexible_logit_ccps, bbl_objective,
    simdata, ergodic, eqb, equilibrium_demand,
    msm_moments, msm_objective,
    print_estimates, print_mc_tables

include("model.jl")       # primitives: utility, transitions, Bellman operators
include("dpsolver.jl")    # SA + Newton–Kantorovich poly-algorithm
include("data.jl")        # bus data, frequency estimator, sufficient statistics
include("nfxp.jl")        # NFXP maximum likelihood
include("mpec.jl")        # MPEC (JuMP + Ipopt) constrained likelihood
include("npl.jl")         # NPL / CCP estimators, Ψ operator, BBL objective
include("simulate.jl")    # simulation, ergodic distribution, demand curve
include("msm.jl")         # method of simulated moments criterion
include("output.jl")      # printed tables

end # module
