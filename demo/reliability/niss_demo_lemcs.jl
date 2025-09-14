# NISS (Non-Intrusive Sensitivity Analysis) Demo
using Pkg
Pkg.activate(".")
using UncertaintyQuantification
using DataFrames
using Distributions
using Random
using Statistics
using Plots

# Random.seed!(42)


function g_function(df::DataFrame)
    n_samples = nrow(df)
    outputs = Vector{Float64}(undef, n_samples)
    
    for i in 1:n_samples
        x1, x2 = df[i, :x1], df[i, :x2]
        outputs[i] = 1.0 - (x1 - 1.0)^2 / 9.0 - (x2 - 1.0)^3 / 16.0
    end
    
    return outputs
end

model = Model(g_function, :output)

inputs = [
    ProbabilityBox{Normal}([
        Interval(-0.2, 0.2, :μ), 
        Interval(0.8, 1.2, :σ)
    ], :x1),
    ProbabilityBox{Normal}([
        Interval(-0.2, 0.2, :μ), 
        Interval(0.8, 1.2, :σ)
    ], :x2)
]

# Define anchor point (parameter values at center/paper values)
anchor = Dict(
    :x1 => [0.0, 1.0],    # [μ1, σ1] for x1
    :x2 => [0.0, 1.0]     # [μ1, σ2] for x2
)

println("\nRunning LEMCS-cut-HDMR analysis...")

# Run LEMCS analysis
lemcs_result = lemcs_cut_hdmr(
    model,
    inputs,
    :output,
    anchor;
    order=2,
    N=5000
)
# println("\nRunning GEMCS-cut-HDMR analysis...")
# gemcs_result = gemcs_rs_hdmr(
#     model,
#     inputs,
#     :output;
#     order=2,
#     N=2000
# )



# E_1, hdmr_variances= lemcs_result(Dict(:x1 => [0.2, 1.0], :x2 => [0.0, 1.0]))
# E_1, hdmr_variances= gemcs_result(Dict(:x1 => [0.2, 1.0], :x2 => [0.0, 1.0]))

println("\nCreating Plots...")

# LEMCS PLOTS
mu_range = range(-0.2, 0.2, length=50)
E1_mu_x1 = zeros(length(mu_range))
E1_mu_x2 = zeros(length(mu_range))
V1_mu_x1 = zeros(length(mu_range))
V1_mu_x2 = zeros(length(mu_range))

for (i, mu) in enumerate(mu_range)
    θ = Dict(:x1 => [mu, 1.0], :x2 => [mu, 1.0])
    E_1, V_1 = lemcs_result(θ)
    E1_mu_x1[i] = get(E_1, [1], NaN)
    V1_mu_x1[i] = get(V_1, [1], NaN)
    E1_mu_x2[i] = get(E_1, [3], NaN)
    V1_mu_x2[i] = get(V_1, [3], NaN)
end

sigma_range = range(0.8, 1.2, length=50)
E1_sigma_x1 = zeros(length(sigma_range))
E1_sigma_x2 = zeros(length(sigma_range))
V1_sigma_x1 = zeros(length(sigma_range))
V1_sigma_x2 = zeros(length(sigma_range))

for (i, sigma) in enumerate(sigma_range)
    θ = Dict(:x1 => [0.0, sigma], :x2 => [0.0, sigma])
    E_1, V_1 = lemcs_result(θ)
    E1_sigma_x1[i] = get(E_1, [2], NaN)
    V1_sigma_x1[i] = get(V_1, [2], NaN)
    E1_sigma_x2[i] = get(E_1, [4], NaN)
    V1_sigma_x2[i] = get(V_1, [4], NaN)
end

lp1 = plot(mu_range, E1_mu_x1, xlabel="μ₁", ylabel="E₁[1]", title="x1 μ component", legend=:topright, ylims=(-0.15, 0.15))
plot!(lp1, mu_range, V1_mu_x1, label="Var[1]", linestyle=:dash, color=:red)
plot!(lp1, label="Mean[1]", color=:blue)

lp2 = plot(mu_range, E1_mu_x2, xlabel="μ₂", ylabel="E₁[3]", title="x2 μ component", legend=:topright, ylims=(-0.15, 0.15))
plot!(lp2, mu_range, V1_mu_x2, label="Var[3]", linestyle=:dash, color=:red)
plot!(lp2, label="Mean[3]", color=:blue)

lp3 = plot(sigma_range, E1_sigma_x1, xlabel="σ₁", ylabel="E₁[2]", title="x1 σ component", legend=:topright, ylims=(-0.15, 0.15))
plot!(lp3, sigma_range, V1_sigma_x1, label="Var[2]", linestyle=:dash, color=:red)
plot!(lp3, label="Mean[2]", color=:blue)

lp4 = plot(sigma_range, E1_sigma_x2, xlabel="σ₂", ylabel="E₁[4]", title="x2 σ component", legend=:topright, ylims=(-0.15, 0.15))
plot!(lp4, sigma_range, V1_sigma_x2, label="Var[4]", linestyle=:dash, color=:red)
plot!(lp4, label="Mean[4]", color=:blue)

lemcs_plot = plot(lp1, lp2, lp3, lp4, layout=(2,2), size=(900,700), title="Single HDMR Comp (LEMCS)")


# # GEMCS PLOTS
# mu_range = range(-0.2, 0.2, length=50)
# E1_mu_x1 = zeros(length(mu_range))
# E1_mu_x2 = zeros(length(mu_range))
# V1_mu_x1 = zeros(length(mu_range))
# V1_mu_x2 = zeros(length(mu_range))

# for (i, mu) in enumerate(mu_range)
#     θ = Dict(:x1 => [mu, 1.0], :x2 => [mu, 1.0])
#     E_1, V_1 = gemcs_result(θ)
#     E1_mu_x1[i] = get(E_1, [1], NaN)
#     V1_mu_x1[i] = get(V_1, [1], NaN)
#     E1_mu_x2[i] = get(E_1, [3], NaN)
#     V1_mu_x2[i] = get(V_1, [3], NaN)
# end

# sigma_range = range(0.8, 1.2, length=50)
# E1_sigma_x1 = zeros(length(sigma_range))
# E1_sigma_x2 = zeros(length(sigma_range))
# V1_sigma_x1 = zeros(length(sigma_range))
# V1_sigma_x2 = zeros(length(sigma_range))

# for (i, sigma) in enumerate(sigma_range)
#     θ = Dict(:x1 => [0.0, sigma], :x2 => [0.0, sigma])
#     E_1, V_1 = gemcs_result(θ)
#     E1_sigma_x1[i] = get(E_1, [2], NaN)
#     V1_sigma_x1[i] = get(V_1, [2], NaN)
#     E1_sigma_x2[i] = get(E_1, [4], NaN)
#     V1_sigma_x2[i] = get(V_1, [4], NaN)
# end

# p1 = plot(mu_range, E1_mu_x1, xlabel="μ₁", ylabel="E₁[1]", title="x1 μ component", legend=:topright, ylims=(-0.15, 0.15))
# plot!(p1, mu_range, V1_mu_x1, label="Var[1]", linestyle=:dash, color=:red)
# plot!(p1, label="Mean[1]", color=:blue)

# p2 = plot(mu_range, E1_mu_x2, xlabel="μ₂", ylabel="E₁[3]", title="x2 μ component", legend=:topright, ylims=(-0.15, 0.15))
# plot!(p2, mu_range, V1_mu_x2, label="Var[3]", linestyle=:dash, color=:red)
# plot!(p2, label="Mean[3]", color=:blue)

# p3 = plot(sigma_range, E1_sigma_x1, xlabel="σ₁", ylabel="E₁[2]", title="x1 σ component", legend=:topright, ylims=(-0.15, 0.15))
# plot!(p3, sigma_range, V1_sigma_x1, label="Var[2]", linestyle=:dash, color=:red)
# plot!(p3, label="Mean[2]", color=:blue)

# p4 = plot(sigma_range, E1_sigma_x2, xlabel="σ₂", ylabel="E₁[4]", title="x2 σ component", legend=:topright, ylims=(-0.15, 0.15))
# plot!(p4, sigma_range, V1_sigma_x2, label="Var[4]", linestyle=:dash, color=:red)
# plot!(p4, label="Mean[4]", color=:blue)

# gemcs_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(900,700), title="Single HDMR Comp GEMCS")

display(lemcs_plot)
# display(gemcs_plot)