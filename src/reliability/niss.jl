# NISS (Non-Intrusive Sensitivity Analysis)

abstract type NISS end

struct LEMCSCutHDMR <: NISS
    anchor::Dict{Symbol, Vector{Float64}}
    inputs::Vector{<:ProbabilityBox}
    output::Symbol
    order::Int
    samples::DataFrame
    sensitivity_indices::Dict{Symbol, Tuple{Float64, Float64}}
end

function LEMCSCutHDMR(
    anchor::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:ProbabilityBox}, 
    output::Symbol, 
    order::Int, 
    samples::DataFrame
)
    return LEMCSCutHDMR(anchor, inputs, output, order, samples, Dict{Symbol, Tuple{Float64, Float64}}())
end

struct GEMCS_RS_HDMR <: NISS
    anchor::Vector{Dict{Symbol, Vector{Float64}}}
    inputs::Vector{<:ProbabilityBox}
    output::Symbol
    order::Int
    samples::DataFrame
    sensitivity_indices::Dict{Symbol, Tuple{Float64, Float64}}
end

function GEMCS_RS_HDMR(
    anchor::Vector{Dict{Symbol, Vector{Float64}}}, 
    inputs::Vector{<:ProbabilityBox}, 
    output::Symbol, 
    order::Int, 
    samples::DataFrame
)
    return GEMCS_RS_HDMR(anchor, inputs, output, order, samples, Dict{Symbol, Tuple{Float64, Float64}}())
end

function (niss::NISS)(θ::Dict{Symbol, Vector{Float64}})
    varnames = [pbox.name for pbox in niss.inputs]
    total_params = sum(length(pbox.parameters) for pbox in niss.inputs)
    sample_matrix = Matrix(niss.samples[:, varnames])
    y_values = niss.samples[:, niss.output]
    N = size(sample_matrix, 1)

    hdmr_components = Dict{Vector{Int}, Float64}()
    hdmr_variances = Dict{Vector{Int}, Float64}()
    
    E_y0 = mean(y_values)
    hdmr_components[Int[]] = E_y0
    hdmr_variances[Int[]] = var(y_values) / N
    
    for current_order in 1:niss.order
        combinations = _combinations(1:total_params, current_order)
        
        for indices in combinations
            estimate, variance = _compute_combination_component(
                sample_matrix, y_values, niss.inputs, indices, niss.anchor, θ
            )
            
            hdmr_components[indices] = estimate
            hdmr_variances[indices] = variance
        end
    end
    
    return hdmr_components, hdmr_variances
end

function lemcs_cut_hdmr(
    model::Model, 
    inputs::Vector{<:ProbabilityBox},
    output::Symbol, 
    anchor::Dict{Symbol, Vector{Float64}}; 
    order::Int=2, 
    N::Int=1000
)
    anchor_inputs = [map_to_precise(anchor[pbox.name], pbox) for pbox in inputs]
    samples = sample(anchor_inputs, N)
    evaluate!(model, samples)

    lemcs_cut_hdmr = LEMCSCutHDMR(anchor, inputs, output, order, samples)
    sensitivity_indices = _compute_sensitivity_indices(lemcs_cut_hdmr)

    return LEMCSCutHDMR(anchor, inputs, output, order, samples, sensitivity_indices)
end
function gemcs_rs_hdmr(
    model::Model, 
    inputs::Vector{<:ProbabilityBox},
    output::Symbol;
    order::Int=2, 
    N::Int=1000
)
    total_params = sum(length(pbox.parameters) for pbox in inputs)
    param_bounds = [(param.lb, param.ub) for pbox in inputs for param in pbox.parameters]

    lhs = [shuffle(collect(0:N-1)) for _ in 1:total_params]
    lhs_points = [(lhs[j][i] + 0.5) / N for i in 1:N, j in 1:total_params]

    anchor = Vector{Dict{Symbol, Vector{Float64}}}(undef, N)
    for i in 1:N
        anchor_point = Dict{Symbol, Vector{Float64}}()
        idx = 1
        for pbox in inputs
            nparams = length(pbox.parameters)
            vals = [param_bounds[idx+j-1][1] + lhs_points[i, idx+j-1] * (param_bounds[idx+j-1][2] - param_bounds[idx+j-1][1]) for j in 1:nparams]
            anchor_point[pbox.name] = vals
            idx += nparams
        end
        anchor[i] = anchor_point
    end

    anchor_samples_list = map(anchor_point -> begin
        anchor_inputs = [map_to_precise(anchor_point[pbox.name], pbox) for pbox in inputs]
        sample(anchor_inputs, 1)
    end, anchor)
    samples = vcat(anchor_samples_list...)
    evaluate!(model, samples)

    gemcs_rs_hdmr = GEMCS_RS_HDMR(anchor, inputs, output, order, samples)
    sensitivity_indices = _compute_sensitivity_indices(gemcs_rs_hdmr)

    return GEMCS_RS_HDMR(anchor, inputs, output, order, samples, sensitivity_indices)
end

function _compute_combination_component(
    sample_matrix::Matrix{Float64}, 
    y_values::Vector{Float64},
    inputs::Vector{<:ProbabilityBox}, 
    combination_indices::Vector{Int}, 
    anchorpoint::Dict{Symbol, Vector{Float64}}, # local - single anchor point
    θ::Dict{Symbol, Vector{Float64}}
)
    N = size(sample_matrix, 1)
    current_order = length(combination_indices)
    
    weighted_values = map(k -> y_values[k] * _compute_importance_ratio(
        sample_matrix[k, :], combination_indices, anchorpoint, θ, inputs, current_order
    ), 1:N)

    mean_val = mean(weighted_values)
    variance = sum(weighted_values.^2 .- N * mean_val^2) / (N * (N - 1))
    
    return mean_val, variance
end

function _compute_combination_component(
    sample_matrix::Matrix{Float64}, 
    y_values::Vector{Float64},
    inputs::Vector{<:ProbabilityBox}, 
    combination_indices::Vector{Int}, 
    anchorpoint::Vector{Dict{Symbol, Vector{Float64}}}, # global - multiple anchor points
    θ::Dict{Symbol, Vector{Float64}}
)
    N = size(sample_matrix, 1)
    current_order = length(combination_indices)

    weighted_values = map(k -> y_values[k] * _compute_importance_ratio(
        sample_matrix[k, :], combination_indices, anchorpoint[k], θ, inputs, current_order
    ), 1:N)

    mean_val = mean(weighted_values)
    variance = sum(weighted_values.^2 .- N * mean_val^2) / (N * (N - 1))
    
    return mean_val, variance
end

function _compute_importance_ratio(  # currently max order 2
    sample_point::Vector{Float64}, 
    combination_indices::Vector{Int}, 
    reference_point::Dict{Symbol, Vector{Float64}}, 
    θ::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:ProbabilityBox},
    order::Int
)
    total_ratio = _compute_conditional_pdf_ratio(sample_point, combination_indices, reference_point, θ, inputs)
    
    for subset_size in 1:(order-1)
        # sign = (-1)^subset_size
        for subset in _combinations(1:order, subset_size) 
            subset_combination_indices = [combination_indices[i] for i in subset]
            total_ratio -= _compute_conditional_pdf_ratio(sample_point, subset_combination_indices, reference_point, θ, inputs)
        end
    end
    
    return total_ratio + (-1)^order  # Check this later if it is inclusion/exclution for higher order or not
end

function _compute_conditional_pdf_ratio(
    sample_point::Vector{Float64}, 
    param_indices::Vector{Int}, 
    reference_point::Dict{Symbol, Vector{Float64}}, 
    θ::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:ProbabilityBox}
)
    modified_params = deepcopy(reference_point)
    global_param_idx = 1
    for pbox in inputs, local_param_idx in 1:length(pbox.parameters)
        if global_param_idx in param_indices
            modified_params[pbox.name][local_param_idx] = θ[pbox.name][local_param_idx]
        end
        global_param_idx += 1
    end

    numerator_pdf = _compute_joint_pdf(sample_point, modified_params, inputs)
    denominator_pdf = _compute_joint_pdf(sample_point, reference_point, inputs)
    
    return denominator_pdf ≈ 0.0 ? 1.0 : numerator_pdf / denominator_pdf
end

function _compute_joint_pdf(     # Independent only
    sample_point::Vector{Float64}, 
    parameter_values::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:ProbabilityBox}
)
    joint_pdf = 1.0
    
    for (i, pbox) in enumerate(inputs)
        precise_rv = map_to_precise(parameter_values[pbox.name], pbox)
        
        individual_pdf = pdf(precise_rv.dist, sample_point[i])
        joint_pdf *= individual_pdf
        
        if individual_pdf ≈ 0.0
            return 0.0
        end
    end
    
    return joint_pdf
end

function _compute_sensitivity_indices(lemcs::LEMCSCutHDMR; Nt::Int=50)
    total_params = sum(length(p.parameters) for p in lemcs.inputs)
    varnames = [pbox.name for pbox in lemcs.inputs]
    sample_matrix = Matrix(lemcs.samples[:, varnames])
    y_values = lemcs.samples[:, lemcs.output]

    param_ranges = Dict{Int, Vector{Float64}}()
    param_names = String[]
    
    count = 0
    for (var_idx, pbox) in enumerate(lemcs.inputs)
        for (param_idx, param) in enumerate(pbox.parameters)
            count += 1
            
            interval = pbox.parameters[param_idx]
            param_min = interval.lb
            param_max = interval.ub
            
            param_ranges[count] = range(param_min, param_max, length=Nt)
            push!(param_names, "$(pbox.name)_$(param.name)")
        end
    end

    function get_component_function(indices::Vector{Int})
        return function(θ_dict::Dict{Symbol, Vector{Float64}})
            estimate, variance = _compute_combination_component(
                sample_matrix, y_values, lemcs.inputs, indices, lemcs.anchor, θ_dict
            )
            return estimate, variance
        end
    end

    component_vars = Dict{Vector{Int}, Float64}()
    component_variance_estimates = Dict{Vector{Int}, Float64}()

    for i in 1:total_params
        component_func = get_component_function([i])
        vals = param_ranges[i]
        
        expectation_values = zeros(Nt)
        variance_values = zeros(Nt)
        
        for k in 1:Nt
            θ_dict = deepcopy(lemcs.anchor)

            param_idx = 0
            for pbox in lemcs.inputs
                for j in 1:length(pbox.parameters)
                    param_idx += 1
                    if param_idx == i
                        θ_dict[pbox.name][j] = vals[k]
                        break
                    end
                end
            end
            
            estimate, variance = component_func(θ_dict)
            expectation_values[k] = estimate
            variance_values[k] = variance
        end
        
        component_vars[[i]] = var(expectation_values)
        component_variance_estimates[[i]] = var(variance_values)
    end

    if lemcs.order >= 2
        for i in 1:total_params
            for j in i+1:total_params
                component_func = get_component_function(sort([i,j]))
                vals_i = param_ranges[i]
                vals_j = param_ranges[j]
                
                coarse_Nt = min(Nt, 20)
                i_indices = round.(Int, range(1, Nt, length=coarse_Nt))
                j_indices = round.(Int, range(1, Nt, length=coarse_Nt))
                
                expectation_values = zeros(coarse_Nt, coarse_Nt)
                variance_values = zeros(coarse_Nt, coarse_Nt)
                
                for (ki, k) in enumerate(i_indices)
                    for (lj, l) in enumerate(j_indices)
                        θ_dict = deepcopy(lemcs.anchor)
                        param_idx = 0
                        for pbox in lemcs.inputs
                            for m in 1:length(pbox.parameters)
                                param_idx += 1
                                if param_idx == i
                                    θ_dict[pbox.name][m] = vals_i[k]
                                elseif param_idx == j
                                    θ_dict[pbox.name][m] = vals_j[l]
                                end
                            end
                        end
                        estimate, variance = component_func(θ_dict)
                        expectation_values[ki, lj] = estimate
                        variance_values[ki, lj] = variance
                    end
                end
                
                component_vars[sort([i,j])] = var(vec(expectation_values))
                component_variance_estimates[sort([i,j])] = var(vec(variance_values))
            end
        end
    end

    total_var = sum(values(component_vars))
    total_var_estimates = sum(values(component_variance_estimates))
    
    sensitivity_indices = Dict{Symbol, Tuple{Float64, Float64}}()
    
    for (indices, variance) in component_vars
        if length(indices) == 1
            param_name = Symbol(param_names[indices[1]])
        else
            names_combined = [param_names[i] for i in indices]
            param_name = Symbol(join(names_combined, "_"))
        end

        se_sensitivity_index = total_var > 0 ? variance / total_var : 0.0
        sv_sensitivity_index = total_var_estimates > 0 ? component_variance_estimates[indices] / total_var_estimates : 0.0
        sensitivity_indices[param_name] = (se_sensitivity_index, sv_sensitivity_index)
    end

    return sensitivity_indices
end

function _compute_sensitivity_indices(gemcs::GEMCS_RS_HDMR)
    total_params = sum(length(p.parameters) for p in gemcs.inputs)
    varnames = [pbox.name for pbox in gemcs.inputs]
    sample_matrix = Matrix(gemcs.samples[:, varnames])
    y_values = gemcs.samples[:, gemcs.output]
    N = size(sample_matrix, 1)

    param_names = String[]
    for (pbox) in gemcs.inputs
        for (param) in pbox.parameters
            push!(param_names, "$(pbox.name)_$(param.name)")
        end
    end

    evals = Dict{Int, Dict{Vector{Int}, Float64}}()
    var_evals = Dict{Int, Dict{Vector{Int}, Float64}}() # later put dict here again
    for k in 1:N
        expected_val, variance = gemcs(gemcs.anchor[k])
        evals[k] = expected_val # sum(values(expected_val))
        var_evals[k] = variance # sum(values(variance))
        (k % 100 == 0 || k == N) && println("Eval $k / $N done.")
    end

    component_vars = Dict{Vector{Int}, Float64}()
    component_variance_estimates = Dict{Vector{Int}, Float64}()

    for i in _combinations(1:total_params, 1)
        expectation_values = [get(evals[k], i, NaN) for k in 1:N]
        variance_values = [get(var_evals[k], i, NaN) for k in 1:N]

        component_vars[i] = mean(expectation_values.^2)
        component_variance_estimates[i] = var(variance_values) # still probably bullshit
    end

    if gemcs.order >= 2
        for i in _combinations(1:total_params, 2)
            expectation_values = [get(evals[k], i, NaN) for k in 1:N]
            variance_values = [get(var_evals[k], i, NaN) for k in 1:N]

            component_vars[i] = mean(expectation_values.^2)
            component_variance_estimates[i] = mean(variance_values) # still probably bullshit
        end
    end

    total_var = var([sum(values(v)) for v in values(evals)])
    total_var_estimates = var(values(component_variance_estimates)) # still bullshit
    
    sensitivity_indices = Dict{Symbol, Tuple{Float64, Float64}}()
    
    for (indices, variance) in component_vars
        if length(indices) == 1
            param_name = Symbol(param_names[indices[1]])
        else
            names_combined = [param_names[i] for i in indices]
            param_name = Symbol(join(names_combined, "_"))
        end

        se_sensitivity_index = total_var > 0 ? variance / total_var : 0.0
        sv_sensitivity_index = total_var_estimates > 0 ? component_variance_estimates[indices] / total_var_estimates : 0.0
        sensitivity_indices[param_name] = (se_sensitivity_index, sv_sensitivity_index)
    end

    return sensitivity_indices
end