# This file is adapted from
# https://bitbucket.org/mvihola/abc-mcmc/src/master/src/ABC.jl
using AdaptiveMCMC
using Random
using LinearAlgebra

function adaptiveabcmcmc(log_prior, sim, n, theta0, b=ceil(Int, n/10))
    d = length(theta0)
    # Initialise random walk sampler state: r.x current state, r.y proposal
    theta = RWMState(theta0)

    y_dist = Inf
    # Ensure that initial distance is finite & postive:
    while !isfinite(y_dist)
        y_dist = sim(theta.x)
    end
    x_dist = y_dist

    tol = y_dist + (y_dist == 0.0)

    tolerance = AdaptiveScalingMetropolis(0.1, 1.0/tol)

    # Initialise Adaptive Metropolis state (with default parameters)
    state = AdaptiveMetropolis(theta0) #, 2.38/sqrt(d))

    Theta = Array{Float64}(undef,d,n)
    Dist = Vector{Float64}(undef,n)
    p_x = log_prior(theta.x)
    for k = 1:(n+b)
        draw!(theta, state)


        p_y = log_prior(theta.y)
        alpha = 0.0
        if p_y != -Inf
            y_dist = sim(theta.y)
            if y_dist < tol
                alpha = min(1.0, exp(p_y - p_x)) # The Metropolis acceptance probability

                if rand() <= alpha
                    p_x = p_y
                    x_dist = y_dist

                    accept!(theta)
                end
            end
        end

        # Do the adaptation update:
        adapt!(state, theta, alpha, k)

        if k <= b
            adapt!(tolerance, theta, alpha, k)
            tol = 1.0/exp(tolerance.sc[])
        else
            Theta[:,k-b] = theta.x   # Save the current sample
            Dist[k-b] = x_dist
        end
    end
    (Theta=Theta, Dist=Dist)
end
