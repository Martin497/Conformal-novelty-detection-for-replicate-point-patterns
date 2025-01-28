using Distributions
using PointPatternStatistics
using SequentialPointProcesses
using Optim
using ComponentArrays
using UnPack
using AdaptiveMCMC
using ProgressMeter
include("readpattern.jl")

# Read point pattern and observation window for one subject
pp = readpattern(20)

# Define model and log posterior
M1(R, kappa) = Softcore(d -> (R/d)^(2/kappa))
M2(R, kappa, theta) = Mixture(M1(R, kappa), SequentialPointProcesses.Uniform(), theta)
function log_posterior(x)
    @unpack R, kappa, theta = x
    # Compute log prior
    lp = 0
    lp += logpdf(Gamma(3, 70/3), R)
    lp += logpdf(Distributions.Uniform(0, 1), kappa)
    lp += logpdf(Distributions.Uniform(0, 1), theta)
    # If log prior is -Inf don't compute likelihood.
    isinf(lp) && return lp
    # Use 120 integration point on x axis.
    # Use multithreading to speed up computation.
    lp + logpdf(M2(R, kappa, theta), pp, SequentialPointProcesses.Options(nx=120, parallel=:threads))
end

# Apply robust adaptive metropolis sampling
# For a more serious analysis N should be 100_000
N = 10_000
Nburnin = div(N, 5)
init = ComponentArray(R=70.0, kappa=0.4, theta=0.1)

begin
    progress = Progress(N+Nburnin, dt=1)
    a = adaptive_rwm(init, x -> begin
        next!(progress)
        log_posterior(x)
    end, N+Nburnin, algorithm=:ram, b=Nburnin+1)
end
# It may take a while...
X = a.X
# Check that traceplots look like hairy caterpillars
using MCMCChains
# using StatsPlots
chains1 = Chains(X', labels(init))
# plot(chains1)

# Generate samples from the Softcore model using the posterior sample
inds = round.(Int, range(1, N, length=2500))
pps = map(inds) do i
    @unpack R, kappa = convert(typeof(init), X[:,i])
    rand(M1(R, kappa), pp.window, length(pp.data))
end

# plot(plot(pp, labels="", title="Data"), plot(pps[1], labels="", title="Simulation"))

# Compute pair correlation functions for all samples and observed pattern
r = 0:500
gsim = [pcf(pp, r)[2:end] for pp in pps]
gobs = pcf(pp, r)[2:end]
r = 1:500
# plot(r, gobs, label="obs")
# plot!(r, mean(gsim), label="mean")

# Compute global envelope from simulated pair correlation functions
(hi, lo, alpha_interval) = globalenvelope(gsim)
# plot!(r, lo, label="lo")
# plot!(r, hi, label="hi")

# Use ggplot2 to plot the envelope
# using RCall
# using DataFrames
# @rlibrary(ggplot2)
# ggplot(DataFrame(r=1:500, hi=hi, lo=lo, obs=gobs), aes(x=:r, ymax=:hi, y=:obs, ymin=:lo)) +
#     geom_ribbon(fill="grey70") + geom_line() + geom_hline(yintercept=1) +
#     ggtitle("Global 95% envelope for pair-correlation function")

# Plot simulated pattern using base R (requires spatstat to be installed)
# R"plot"(pps[1], main="Simulated pattern")
