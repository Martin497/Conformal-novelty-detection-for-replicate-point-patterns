using Distributions
using AdaptiveMCMC
using ComponentArrays
using UnPack
using PointPatternStatistics
using CSV
using ProgressMeter
include("readpattern.jl")
include("abc_mcmc_sampler.jl")

function log_prior(x)
    @unpack R, p, σ = x
    if R < 40 return -Inf end
    lp = 0
    lp += logpdf(Gamma(3, 10/3), σ)
    lp += logpdf(Distributions.Uniform(0.1, 1), p)
    lp
end

# Helper function for inverses of point pattern summaries
# Find the location r that satisfies v(r) == y by linear interpolation
function summinv(r, v, y)
    i = findfirst(>=(y), v)
    if i == nothing
        return Inf
    end
    # Linear interpolation between (r,v)[i-1] and (r,v)[i]
    y1 = v[i-1]
    y2 = v[i]
    r[i-1] + (r[i] - r[i-1]) * (y - y1)/(y2 - y1)
end

# Compute summaries used in ABC
function summ(pp)
    rF = range(0, 400, length=500)
    vF = Fest(pp, rF)
    rpcf = range(0, 200, length=500)
    vpcf = pcf(pp, rpcf)
    # Set the first 10 pixels to zero in order to discard them
    vpcf[1:26] .= 0.0

    [summinv(rpcf, vpcf, 0.75), summinv(rpcf, vpcf, 1.0), summinv(rF, vF, 0.5)]
end

# Simulate a point pattern from the generative model.
function sim(theta, window)
    @unpack R, p, σ = theta
    X1 = rand(SimpleSequentialInhibition(window, R), T=10^30)
    X2 = map(X1.data) do (x, y)
        (x + σ*randn(), y + σ*randn())
    end
    X3 = filter(X2) do _
        rand() <= p
    end
    PointPattern(filter(inside(window), X3), window)
end

# Simulate a point pattern with given parameters and compute summmaries and
# compute distance between summaries and observed summaries
function sim_dist(theta, pp, s_obs)
    # Simulate a point pattern
    x = sim(theta, pp.window)

    # If there are less than 10 points, reject.
    if length(x) < 10
        return Inf
    end
    # Compute summaries for simulated pattern
    s_sim = summ(x)
    # Compute distance between observed and simulated summary
    norm(s_sim - s_obs)
end

subject_id = 205
pp = readpattern(subject_id)

s_obs = summ(pp)

# N = 10_000
N = 100_000
# N = 1_000_000
init = ComponentArray(R=70.0, p=0.9, σ=10.0)
begin
    # Progress prints a progress bar
    progress = Progress(N, dt=1)
    # Run adaptive ABC-MCMC
    res = adaptiveabcmcmc(log_prior, theta -> begin
        next!(progress)
        sim_dist(theta, pp, s_obs)
    end, N, init)
end

using MCMCChains
#using StatsPlots

X = res.Theta
chain = Chains(X', labels(init))
# Plot chain with an initial tolerance
#plot(chain)

# Take the N/10 or 250 000 (whichever is lower) observations with lowest tolerance
inds = sort(partialsortperm(res.Dist, 1:min(div(N,10), 250_000)))
#plot(chain[inds,:,:])

# Simulate point patterns from the posterior predictive distribution
pps = map(inds) do ind
    x = convert(typeof(init), X[:,ind])
    sim(x, pp.window)
end

for j = 1:length(pps)
    sim_nr = j
    io = open("simulations/gm_subject_$(subject_id)_sim_$sim_nr.txt", "w")
    for i = 1:length(pps[j].data)
        write(io, "$(pps[j].data[i])\n")
    end
    close(io)
end

# Plot observed and simulated point patterns
#plot(plot(pp, labels="", title="Data"), plot(pps[1], labels="", title="Simulation"))

# Compute pair correlation functions for all samples and observed pattern
#r = 0:500
#gsim = [pcf(pp, r)[2:end] for pp in pps]
#gobs = pcf(pp, r)[2:end]
#r = 1:500
#plot(r, gobs, label="obs")
#plot!(r, mean(gsim), label="mean")

# Compute global envelope simulated pair correlation functions
#(hi, lo, alpha_interval) = globalenvelope(gsim)
#plot!(r, lo, label="lo")
#plot!(r, hi, label="hi")

# Use ggplot2 to plot the envelope
#using RCall
#using DataFrames
#@rlibrary(ggplot2)
#ggplot(DataFrame(r=1:500, hi=hi, lo=lo, obs=gobs), aes(x=:r, ymax=:hi, y=:obs, ymin=:lo)) +
#    geom_ribbon(fill="grey70") + geom_line() + geom_hline(yintercept=1) +
#    ggtitle("Global 95% envelope for pair-correlation function")

# Plot simulated pattern using base R (requires spatstat to be installed)
#R"plot"(pps[1], main="Simulated pattern")
