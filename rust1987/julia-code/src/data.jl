# ==========================================================================
# data.jl — Rust's bus data and the frequency estimator for mileage moves
#
# Julia translation of zurcher.readbusdata and the `tabulate`-based
# first-step estimator used in nfxp.estim / run scripts.
# ==========================================================================

"""
    BusData

Estimation sample (one row per bus-month):
- `x`   : discretized mileage state, 1…n
- `d`   : replacement indicator (true = replace)
- `dx1` : mileage increment between t and t+1 in grid cells (0, 1, 2, …)
"""
struct BusData
    x::Vector{Int}
    d::Vector{Bool}
    dx1::Vector{Int}
end

nobs(data::BusData) = length(data.x)

const BUSDATA_CSV = joinpath(@__DIR__, "..", "data", "busdata1234.csv")

"""
    readbusdata(m::Model; csvpath=BUSDATA_CSV, bustypes=[1,2,3,4]) -> BusData

Read Rust's (1987) bus data (converted from `busdata1234.mat`) and build the
estimation sample, replicating `zurcher.readbusdata` step by step:

1. keep the selected bus types,
2. lead the lagged replacement dummy to get d_t,
3. discretize the odometer into `1, …, m.n` cells of width `maxmiles*1000/n`,
4. first-difference the discretized odometer (using x_t itself in
   replacement months, when the odometer was reset),
5. drop each bus's first observation (no lagged mileage).
"""
function readbusdata(m::Model; csvpath::AbstractString=BUSDATA_CSV, bustypes=[1, 2, 3, 4])
    df = CSV.read(csvpath, DataFrame)
    df = df[in.(df.bustype, Ref(Set(bustypes))), :]

    id = Vector{Int}(df.id)
    d1 = Vector{Int}(df.d_lag)                       # lagged replacement dummy d_{t-1}
    d = [d1[2:end]; 0]                               # replacement dummy d_t

    x = ceil.(Int, df.odometer .* m.n ./ (m.maxmiles * 1000))   # discretize odometer

    dx1 = x .- [0; x[1:end-1]]                       # first difference …
    @. dx1 = dx1 * (1 - d1) + x * d1                 # … but x_t itself right after replacement

    same_bus = id .== [0; id[1:end-1]]               # false on each bus's first row
    return BusData(x[same_bus], Bool.(d[same_bus]), dx1[same_bus])
end

"""
    freq_transition(data::BusData) -> Vector{Float64}

First-step frequency estimator of the free mileage-increment probabilities:
the sample shares of `dx1 = 0, 1, …, J-1`, dropping the largest observed
increment (its probability is implied). Mirrors the MATLAB
`tabulate(data.dx1)` / drop-last construction.
"""
function freq_transition(data::BusData)
    maxdx = maximum(data.dx1)
    cnt = zeros(Int, maxdx + 1)
    for dx in data.dx1
        cnt[dx+1] += 1
    end
    all(>(0), cnt) ||
        @warn "some intermediate mileage increments never occur; transition probabilities of zero-count cells will be estimated as 0"
    return cnt[1:end-1] ./ nobs(data)
end

"""
    ChoiceCounts(data, n, J)

Sufficient statistics for the likelihood, aggregated by state. The
log-likelihood and its gradient only depend on the data through

- `keep[x]`, `repl[x]` : number of keep/replace decisions observed at state x,
- `dx[j]`              : number of mileage increments of size j-1,

so they can be evaluated in O(n) instead of O(N). (The MATLAB code loops
over all N·T observations on every likelihood call.)
"""
struct ChoiceCounts
    keep::Vector{Int}
    repl::Vector{Int}
    dx::Vector{Int}
    N::Int
end

function ChoiceCounts(data::BusData, n::Int, J::Int)
    keep = zeros(Int, n); repl = zeros(Int, n); dx = zeros(Int, J)
    for i in eachindex(data.x)
        if data.d[i]
            repl[data.x[i]] += 1
        else
            keep[data.x[i]] += 1
        end
        dx[data.dx1[i]+1] += 1
    end
    return ChoiceCounts(keep, repl, dx, nobs(data))
end
