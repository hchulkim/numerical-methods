# ==========================================================================
# runtests.jl — Correctness checks for the Julia translation
#
# 1. EV- and IV-space fixed points agree (ev = P_keep * V).
# 2. Analytic likelihood gradient matches central finite differences
#    (both partial and full MLE parameterizations).
# 3. NPL pseudo-likelihood gradient matches finite differences.
# 4. The Ψ fixed point reproduces the Bellman fixed-point CCPs.
# 5. Equilibrium distribution is a proper stationary distribution.
#
# Run:  julia test/runtests.jl
# ==========================================================================

include(joinpath(@__DIR__, "..", "src", "Zurcher.jl"))
using .Zurcher
using Test
using LinearAlgebra
using Random

m = Model()                       # Rust's Table X parameters
data = readbusdata(m)
p̂ = freq_transition(data)
mdata = update(m; p=p̂)

@testset "fixed point: EV vs IV space" begin
    ev, pkev, Γev, info_ev = solve(mdata; space=:ev)
    V, pkiv, Γiv, info_iv = solve(mdata; space=:iv)
    @test info_ev.converged
    @test info_iv.converged
    @test maximum(abs, iv_to_ev(Γiv, V) .- ev) < 1e-6
    @test maximum(abs, pkev .- pkiv) < 1e-8
    @test all(0 .< pkev .< 1)
end

function fd_gradient(f, θ; h=1e-6)
    g = similar(θ)
    for k in eachindex(θ)
        θp = copy(θ); θp[k] += h
        θm = copy(θ); θm[k] -= h
        g[k] = (f(θp) - f(θm)) / (2h)
    end
    return g
end

@testset "NFXP gradient vs finite differences" begin
    cnt = ChoiceCounts(data, mdata.n, length(p̂) + 1)
    for (est_p, θ) in [(false, [11.0, 2.3]),
                       (true, [11.0, 2.3, p̂...])]
        cache = NFXPCache(mdata.n)
        feval(t) = nfxp_fgh!(0.0, nothing, nothing, t, mdata, data, cnt,
                             NFXPCache(mdata.n); est_p=est_p)
        G = zeros(length(θ)); H = zeros(length(θ), length(θ))
        f = nfxp_fgh!(0.0, G, H, θ, mdata, data, cnt, cache; est_p=est_p)
        Gfd = fd_gradient(feval, θ)
        @test maximum(abs, G .- Gfd) / max(1.0, maximum(abs, Gfd)) < 1e-5
        @test issymmetric(H) || maximum(abs, H .- H') < 1e-12
        @test isposdef(Symmetric(H))
    end
end

@testset "NPL gradient vs finite differences" begin
    pk0 = flexible_logit_ccps(data, mdata)
    cnt = ChoiceCounts(data, mdata.n, length(p̂) + 1)
    Flu = policy_lu(mdata, pk0)
    pk = similar(pk0); buf = similar(pk0)
    θ = [10.0, 2.0]
    feval(t) = npl_fgh!(0.0, nothing, nothing, t, mdata, data, cnt, pk0, Flu, pk, buf)
    G = zeros(2); H = zeros(2, 2)
    npl_fgh!(0.0, G, H, θ, mdata, data, cnt, pk0, Flu, pk, buf)
    Gfd = fd_gradient(feval, θ)
    @test maximum(abs, G .- Gfd) / max(1.0, maximum(abs, Gfd)) < 1e-5
end

@testset "Ψ fixed point = Bellman fixed point" begin
    _, pk_fxp, _, _ = solve(mdata; space=:ev)
    pk_psi, niter = solve_npl(mdata, fill(0.95, mdata.n))
    @test maximum(abs, pk_psi .- pk_fxp) < 1e-6
end

@testset "equilibrium distribution" begin
    _, pk, _, _ = solve(mdata; space=:ev)
    pp, pp_K, pp_R = eqb(mdata, pk)
    @test abs(sum(pp) - 1.0) < 1e-10
    @test all(pp .> -1e-12)
    Π = Matrix(policy_matrix(mdata, pk))
    @test maximum(abs, Π' * pp .- pp) < 1e-10        # stationarity
    @test maximum(abs, pp_K .+ pp_R .- pp) < 1e-12
end

@testset "MPEC agrees with NFXP" begin
    m0 = Model(RC=0.0, c=0.0)
    res_nfxp = estim_nfxp(data, m0; est_p=false)
    res_mpec = estim_mpec(data, m0)
    @test res_mpec.converged
    @test maximum(abs, res_mpec.θ .- res_nfxp.θ) < 1e-3
    @test abs(res_mpec.loglike - res_nfxp.loglike) < 1e-4
end

@testset "simulation roundtrip" begin
    _, pk, _, _ = solve(mdata; space=:ev)
    sim = simdata(200, 200, mdata, pk; rng=Xoshiro(42))
    sdata = BusData(sim)
    p_sim = freq_transition(sdata)
    @test maximum(abs, p_sim .- pfull(mdata)[1:length(p_sim)]) < 0.02
end

println("All tests passed.")
