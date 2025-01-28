using PointPatternStatistics
# using Plots
include("readpattern.jl")

# Read pattern of subject 20
pp = readpattern(20)
# plot(pp)
g = pcf(pp, 0:500)
# plot(0:500, g)
print(g)

# Use R and spatstat to estimate and plot point pattern summaries
# using RCall
# spatstat = rimport("spatstat")
# R"plot"(pp, main="Observed activated sweat glands")
# R"plot"(spatstat.pcf(pp), main="Observed pair-correlation function")
