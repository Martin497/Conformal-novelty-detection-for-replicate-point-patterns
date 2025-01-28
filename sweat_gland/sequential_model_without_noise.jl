using PointPatternStatistics
using SequentialPointProcesses
using Distributions
using Optim
using ComponentArrays
using UnPack
using Plots
include("readpattern.jl")

# Read point pattern and observation window for subject 20
pp = readpattern(20)

# Define model and loglikelihood
M1(R, kappa) = Softcore(d -> (R/d)^(2/kappa))
function log_likelihood(x)
    @unpack R, kappa = x
    R <= 0.0 && return -Inf
    0.0 <= kappa <= 1.0 || return -Inf
    logpdf(M1(R, kappa), pp, 120)
end

# Maximize loglikelihood
o1 = maximize(log_likelihood, ComponentVector(R=70.0, kappa=0.4))
p1 = Optim.maximizer(o1)

# Generate samples from fitted model
pps = [rand(M1(p1.R, p1.kappa), pp.window, length(pp.data)) for _ in 1:1000]
plot(plot(pp, labels="", title="Data"), plot(pps[1], labels="", title="Simulation"))

# Compute pair correlation functions for all samples and observed pattern
r = 0:500
gsim = [pcf(pp, r)[2:end] for pp in pps]
gobs = pcf(pp, r)[2:end]
r = 1:500
plot(r, gobs, label="obs")
plot!(r, mean(gsim), label="mean")

# Compute global envelope from the simulated pair correlation functions
(hi, lo, alpha_interval) = globalenvelope(gsim)
plot!(r, lo, label="lo")
plot!(r, hi, label="hi")

# Use ggplot2 to plot the envelope
using RCall
using DataFrames
@rlibrary(ggplot2)
ggplot(DataFrame(r=1:500, hi=hi, lo=lo, obs=gobs), aes(x=:r, ymax=:hi, y=:obs, ymin=:lo)) +
    geom_ribbon(fill="grey70") + geom_line() + geom_hline(yintercept=1) +
    ggtitle("Global 95% envelope for pair-correlation function")

# Plot simulated pattern using base R (requires spatstat to be installed)
R"plot"(pps[1], main="Simulated pattern")
