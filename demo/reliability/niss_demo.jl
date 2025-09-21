# NISS (Non-Intrusive Sensitivity Analysis) demo toy example from paper
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
    order=1,
    N = LatinHypercubeSampling(5000)
)
println("\nRunning GEMCS-cut-HDMR analysis...")
gemcs_result = gemcs_rs_hdmr(
    model,
    inputs,
    :output;
    order=1,
    N = 5000
)

# E_1, hdmr_variances= lemcs_result(Dict(:x1 => [0.2, 1.0], :x2 => [0.0, 1.0]))
# E_1, hdmr_variances= gemcs_result(Dict(:x1 => [0.2, 1.0], :x2 => [0.0, 1.0]))

println("\nCreating Plots...")

# LEMCS PLOTS
mu_range = range(-0.2, 0.2, length=20)
E1_mu_x1 = zeros(length(mu_range)); E1_mu_x2 = zeros(length(mu_range))
V1_mu_x1 = zeros(length(mu_range)); V1_mu_x2 = zeros(length(mu_range))
E1_mu_x1_var = zeros(length(mu_range)); E1_mu_x2_var = zeros(length(mu_range))
V1_mu_x1_var = zeros(length(mu_range)); V1_mu_x2_var = zeros(length(mu_range))
for (i, mu) in enumerate(mu_range)
    θ = Dict(:x1 => [mu, 1.0], :x2 => [mu, 1.0])
    E_i, E_i_var, V_i, V_i_var = lemcs_result(θ; detailed=true)
    E1_mu_x1[i] = get(E_i, [1], NaN)
    E1_mu_x1_var[i] = get(E_i_var, [1], NaN)
    V1_mu_x1[i] = get(V_i, [1], NaN)
    V1_mu_x1_var[i] = get(V_i_var, [1], NaN)
    E1_mu_x2[i] = get(E_i, [3], NaN)
    E1_mu_x2_var[i] = get(E_i_var, [3], NaN)
    V1_mu_x2[i] = get(V_i, [3], NaN)
    V1_mu_x2_var[i] = get(V_i_var, [3], NaN)
end

sigma_range = range(0.8, 1.2, length=20)
E1_sigma_x1 = zeros(length(sigma_range)); E1_sigma_x2 = zeros(length(sigma_range))
V1_sigma_x1 = zeros(length(sigma_range)); V1_sigma_x2 = zeros(length(sigma_range))
E1_sigma_x1_var = zeros(length(sigma_range)); E1_sigma_x2_var = zeros(length(sigma_range))
V1_sigma_x1_var = zeros(length(sigma_range)); V1_sigma_x2_var = zeros(length(sigma_range))
for (i, sigma) in enumerate(sigma_range)
    θ = Dict(:x1 => [0.0, sigma], :x2 => [0.0, sigma])
    E_i, E_i_var, V_i, V_i_var = lemcs_result(θ; detailed=true)
    E1_sigma_x1[i] = get(E_i, [2], NaN)
    E1_sigma_x1_var[i] = get(E_i_var, [2], NaN)
    V1_sigma_x1[i] = get(V_i, [2], NaN)
    V1_sigma_x1_var[i] = get(V_i_var, [2], NaN)
    E1_sigma_x2[i] = get(E_i, [4], NaN)
    E1_sigma_x2_var[i] = get(E_i_var, [4], NaN)
    V1_sigma_x2[i] = get(V_i, [4], NaN)
    V1_sigma_x2_var[i] = get(V_i_var, [4], NaN)
end

α = 0.05
z = quantile(Normal(), 1 - α/2)  # ≈ 1.96 for 95% CI

# --- Mean plots with CI ---
ci1_lower = E1_mu_x1 .- z .* sqrt.(max.(E1_mu_x1_var, 0.0))
ci1_upper = E1_mu_x1 .+ z .* sqrt.(max.(E1_mu_x1_var, 0.0))
lp1 = plot(mu_range, E1_mu_x1, ribbon=(E1_mu_x1 - ci1_lower, ci1_upper - E1_mu_x1),
    xlabel="μ₁", ylabel="E_cut_1", title="x1 μ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_cut_1 ± 95% CI")

ci2_lower = E1_mu_x2 .- z .* sqrt.(max.(E1_mu_x2_var, 0.0))
ci2_upper = E1_mu_x2 .+ z .* sqrt.(max.(E1_mu_x2_var, 0.0))
lp2 = plot(mu_range, E1_mu_x2, ribbon=(E1_mu_x2 - ci2_lower, ci2_upper - E1_mu_x2),
    xlabel="μ₂", ylabel="E_cut_3", title="x2 μ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_cut_3 ± 95% CI")

ci3_lower = E1_sigma_x1 .- z .* sqrt.(max.(E1_sigma_x1_var, 0.0))
ci3_upper = E1_sigma_x1 .+ z .* sqrt.(max.(E1_sigma_x1_var, 0.0))
lp3 = plot(sigma_range, E1_sigma_x1, ribbon=(E1_sigma_x1 - ci3_lower, ci3_upper - E1_sigma_x1),
    xlabel="σ₁", ylabel="E_cut_2", title="x1 σ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_cut_2 ± 95% CI")

ci4_lower = E1_sigma_x2 .- z .* sqrt.(max.(E1_sigma_x2_var, 0.0))
ci4_upper = E1_sigma_x2 .+ z .* sqrt.(max.(E1_sigma_x2_var, 0.0))
lp4 = plot(sigma_range, E1_sigma_x2, ribbon=(E1_sigma_x2 - ci4_lower, ci4_upper - E1_sigma_x2),
    xlabel="σ₂", ylabel="E_cut_4", title="x2 σ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_cut_4 ± 95% CI")

lemcs_mean_plot = plot(lp1, lp2, lp3, lp4, layout=(2,2), size=(900,700), title="LEMCS Mean with 95% CI")

# --- Second moment plots with CI ---
ci1_lower2 = V1_mu_x1 .- z .* sqrt.(max.(V1_mu_x1_var, 0.0))
ci1_upper2 = V1_mu_x1 .+ z .* sqrt.(max.(V1_mu_x1_var, 0.0))
lp1_2 = plot(mu_range, V1_mu_x1, ribbon=(V1_mu_x1 - ci1_lower2, ci1_upper2 - V1_mu_x1),
    xlabel="μ₁", ylabel="V_cut_1", title="x1 μ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_cut_1 ± 95% CI")

ci2_lower2 = V1_mu_x2 .- z .* sqrt.(max.(V1_mu_x2_var, 0.0))
ci2_upper2 = V1_mu_x2 .+ z .* sqrt.(max.(V1_mu_x2_var, 0.0))
lp2_2 = plot(mu_range, V1_mu_x2, ribbon=(V1_mu_x2 - ci2_lower2, ci2_upper2 - V1_mu_x2),
    xlabel="μ₂", ylabel="V_cut_3", title="x2 μ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_cut_3 ± 95% CI")

ci3_lower2 = V1_sigma_x1 .- z .* sqrt.(max.(V1_sigma_x1_var, 0.0))
ci3_upper2 = V1_sigma_x1 .+ z .* sqrt.(max.(V1_sigma_x1_var, 0.0))
lp3_2 = plot(sigma_range, V1_sigma_x1, ribbon=(V1_sigma_x1 - ci3_lower2, ci3_upper2 - V1_sigma_x1),
    xlabel="σ₁", ylabel="V_cut_2", title="x1 σ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_cut_2 ± 95% CI")

ci4_lower2 = V1_sigma_x2 .- z .* sqrt.(max.(V1_sigma_x2_var, 0.0))
ci4_upper2 = V1_sigma_x2 .+ z .* sqrt.(max.(V1_sigma_x2_var, 0.0))
lp4_2 = plot(sigma_range, V1_sigma_x2, ribbon=(V1_sigma_x2 - ci4_lower2, ci4_upper2 - V1_sigma_x2),
    xlabel="σ₂", ylabel="V_cut_4", title="x2 σ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_cut_4 ± 95% CI")

lemcs_secondmoment_plot = plot(lp1_2, lp2_2, lp3_2, lp4_2, layout=(2,2), size=(900,700), title="LEMCS Second Moment with 95% CI")

display(lemcs_mean_plot)
display(lemcs_secondmoment_plot)

# GEMCS PLOTS
mu_range = range(-0.2, 0.2, length=20)
E1_mu_x1 = zeros(length(mu_range)); E1_mu_x2 = zeros(length(mu_range))
V1_mu_x1 = zeros(length(mu_range)); V1_mu_x2 = zeros(length(mu_range))
E1_mu_x1_var = zeros(length(mu_range)); E1_mu_x2_var = zeros(length(mu_range))
V1_mu_x1_var = zeros(length(mu_range)); V1_mu_x2_var = zeros(length(mu_range))
for (i, mu) in enumerate(mu_range)
    θ = Dict(:x1 => [mu, 1.0], :x2 => [mu, 1.0])
    E_i, E_i_var, V_i, V_i_var = gemcs_result(θ; detailed=true)
    E1_mu_x1[i] = get(E_i, [1], NaN)
    E1_mu_x1_var[i] = get(E_i_var, [1], NaN)
    V1_mu_x1[i] = get(V_i, [1], NaN)
    V1_mu_x1_var[i] = get(V_i_var, [1], NaN)
    E1_mu_x2[i] = get(E_i, [3], NaN)
    E1_mu_x2_var[i] = get(E_i_var, [3], NaN)
    V1_mu_x2[i] = get(V_i, [3], NaN)
    V1_mu_x2_var[i] = get(V_i_var, [3], NaN)
end

sigma_range = range(0.8, 1.2, length=20)
E1_sigma_x1 = zeros(length(sigma_range)); E1_sigma_x2 = zeros(length(sigma_range))
V1_sigma_x1 = zeros(length(sigma_range)); V1_sigma_x2 = zeros(length(sigma_range))
E1_sigma_x1_var = zeros(length(sigma_range)); E1_sigma_x2_var = zeros(length(sigma_range))
V1_sigma_x1_var = zeros(length(sigma_range)); V1_sigma_x2_var = zeros(length(sigma_range))
for (i, sigma) in enumerate(sigma_range)
    θ = Dict(:x1 => [0.0, sigma], :x2 => [0.0, sigma])
    E_i, E_i_var, V_i, V_i_var = gemcs_result(θ; detailed=true)
    E1_sigma_x1[i] = get(E_i, [2], NaN)
    E1_sigma_x1_var[i] = get(E_i_var, [2], NaN)
    V1_sigma_x1[i] = get(V_i, [2], NaN)
    V1_sigma_x1_var[i] = get(V_i_var, [2], NaN)
    E1_sigma_x2[i] = get(E_i, [4], NaN)
    E1_sigma_x2_var[i] = get(E_i_var, [4], NaN)
    V1_sigma_x2[i] = get(V_i, [4], NaN)
    V1_sigma_x2_var[i] = get(V_i_var, [4], NaN)
end

α = 0.05
z = quantile(Normal(), 1 - α/2)  # ≈ 1.96 for 95% CI

# --- Mean plots with CI ---
ci1_lower = E1_mu_x1 .- z .* sqrt.(max.(E1_mu_x1_var, 0.0))
ci1_upper = E1_mu_x1 .+ z .* sqrt.(max.(E1_mu_x1_var, 0.0))
lp1 = plot(mu_range, E1_mu_x1, ribbon=(E1_mu_x1 - ci1_lower, ci1_upper - E1_mu_x1),
    xlabel="μ₁", ylabel="E_RS_1", title="x1 μ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_RS_1 ± 95% CI")

ci2_lower = E1_mu_x2 .- z .* sqrt.(max.(E1_mu_x2_var, 0.0))
ci2_upper = E1_mu_x2 .+ z .* sqrt.(max.(E1_mu_x2_var, 0.0))
lp2 = plot(mu_range, E1_mu_x2, ribbon=(E1_mu_x2 - ci2_lower, ci2_upper - E1_mu_x2),
    xlabel="μ₂", ylabel="E_RS_3", title="x2 μ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_RS_3 ± 95% CI")

ci3_lower = E1_sigma_x1 .- z .* sqrt.(max.(E1_sigma_x1_var, 0.0))
ci3_upper = E1_sigma_x1 .+ z .* sqrt.(max.(E1_sigma_x1_var, 0.0))
lp3 = plot(sigma_range, E1_sigma_x1, ribbon=(E1_sigma_x1 - ci3_lower, ci3_upper - E1_sigma_x1),
    xlabel="σ₁", ylabel="E_RS_2", title="x1 σ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_RS_2 ± 95% CI")

ci4_lower = E1_sigma_x2 .- z .* sqrt.(max.(E1_sigma_x2_var, 0.0))
ci4_upper = E1_sigma_x2 .+ z .* sqrt.(max.(E1_sigma_x2_var, 0.0))
lp4 = plot(sigma_range, E1_sigma_x2, ribbon=(E1_sigma_x2 - ci4_lower, ci4_upper - E1_sigma_x2),
    xlabel="σ₂", ylabel="E_RS_4", title="x2 σ mean", legend=:topright, ylims=(-0.15, 0.15), color=:blue, label="E_RS_4 ± 95% CI")

gemcs_mean_plot = plot(lp1, lp2, lp3, lp4, layout=(2,2), size=(900,700), title="GEMCS Mean with 95% CI")

# --- Second moment plots with CI ---
ci1_lower2 = V1_mu_x1 .- z .* sqrt.(max.(V1_mu_x1_var, 0.0))
ci1_upper2 = V1_mu_x1 .+ z .* sqrt.(max.(V1_mu_x1_var, 0.0))
lp1_2 = plot(mu_range, V1_mu_x1, ribbon=(V1_mu_x1 - ci1_lower2, ci1_upper2 - V1_mu_x1),
    xlabel="μ₁", ylabel="V_RS_1", title="x1 μ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_RS_1 ± 95% CI")

ci2_lower2 = V1_mu_x2 .- z .* sqrt.(max.(V1_mu_x2_var, 0.0))
ci2_upper2 = V1_mu_x2 .+ z .* sqrt.(max.(V1_mu_x2_var, 0.0))
lp2_2 = plot(mu_range, V1_mu_x2, ribbon=(V1_mu_x2 - ci2_lower2, ci2_upper2 - V1_mu_x2),
    xlabel="μ₂", ylabel="V_RS_3", title="x2 μ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_RS_3 ± 95% CI")

ci3_lower2 = V1_sigma_x1 .- z .* sqrt.(max.(V1_sigma_x1_var, 0.0))
ci3_upper2 = V1_sigma_x1 .+ z .* sqrt.(max.(V1_sigma_x1_var, 0.0))
lp3_2 = plot(sigma_range, V1_sigma_x1, ribbon=(V1_sigma_x1 - ci3_lower2, ci3_upper2 - V1_sigma_x1),
    xlabel="σ₁", ylabel="V_RS_2", title="x1 σ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_RS_2 ± 95% CI")

ci4_lower2 = V1_sigma_x2 .- z .* sqrt.(max.(V1_sigma_x2_var, 0.0))
ci4_upper2 = V1_sigma_x2 .+ z .* sqrt.(max.(V1_sigma_x2_var, 0.0))
lp4_2 = plot(sigma_range, V1_sigma_x2, ribbon=(V1_sigma_x2 - ci4_lower2, ci4_upper2 - V1_sigma_x2),
    xlabel="σ₂", ylabel="V_RS_4", title="x2 σ second moment", legend=:topright, ylims=(-0.30, 0.65), color=:green, label="V_RS_4 ± 95% CI")

gemcs_secondmoment_plot = plot(lp1_2, lp2_2, lp3_2, lp4_2, layout=(2,2), size=(900,700), title="GEMCS Second Moment with 95% CI")

display(gemcs_mean_plot)
display(gemcs_secondmoment_plot)