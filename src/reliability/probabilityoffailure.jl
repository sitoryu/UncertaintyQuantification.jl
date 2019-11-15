function probability_of_failure(
    models::Union{Array{<:AbstractModel},AbstractModel},
    performance::Function,
    inputs::Union{Array{<:AbstractInput},AbstractInput},
    sim::MonteCarlo,
)

    samples = sample(inputs, sim.n)

    # Models
    for m in models
        samples = evaluate(m, samples)
    end

    # Probability of failure
    pf = sum(performance(samples) .< 0) / sim.n

    return pf, samples
end

function probability_of_failure(
    models::Union{Array{<:AbstractModel},AbstractModel},
    performance::Function,
    inputs::Union{Array{<:AbstractInput},AbstractInput},
    sim::LineSampling,
)
    samples = sample(inputs, sim)

    for m in models
        samples = evaluate(m, samples)
    end

    p = reshape(performance(samples), length(sim.points), sim.lines)

    ϕ = Normal()
    pf = 0
    roots_found = 0
    x = median(sim.points)
    for i = 1:sim.lines
        if all(p[:, i] .< 0)
            @warn "All samples for line $i are inside the failure domain"
            continue
        elseif all(p[:, i] .> 0)
            @warn "All samples for line $i are outside the failure domain"
            continue
        end
        spl = Spline1D(sim.points, p[:, i])
        try
            root = Dierckx.roots(spl)[1]
            pf += cdf.(ϕ, -1 * root)
            roots_found += 1
        catch e
            @warn "Intersection with failure domain not found for line $i ($e)"
        end
    end

    pf /= roots_found

    return pf, samples
end
