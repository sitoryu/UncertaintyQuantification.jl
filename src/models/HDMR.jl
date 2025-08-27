# High level Model Representation

struct HDMRRepresentation
    anchor::Vector{Float64}
    f0::Float64
    f1::Vector{Function}
    f2::Dict{Tuple{Int,Int},Function}
    inputs::Vector{RandomVariable}
    output::Symbol
end

function (hdmr::HDMRRepresentation)(h::Vector{Float64})
    val = hdmr.f0
    d = length(hdmr.anchor)
    for i in 1:d
        val += hdmr.f1[i](h[i])
    end
    if !isempty(hdmr.f2)
        for ((i, j), g) in hdmr.f2
            val += g(h[i], h[j])
        end
    end
    return val
end

function cut_HDMR(model::Model, inputs::Vector{RandomVariable}, output::Symbol, anchor::DataFrame; order::Int=2, degree::Int=3, samples::Int=50)
    varnames = names(anchor)
    d = length(varnames)

    anchor_vec = [anchor[1, name] for name in varnames]
    
    # f0 at anchor
    anchor_copy = deepcopy(anchor)
    evaluate!(model, anchor_copy)
    f0 = anchor_copy[1, output]

    # First Order
    f1 = Vector{Any}(undef, d)
    xs = range(0.0, 1.0; length=samples)
    for i in 1:d
        X = zeros(samples, degree+1)
        y = zeros(samples)
        for (k, xi) in enumerate(xs)
            h = deepcopy(anchor)
            h[!, varnames[i]] .= xi
            evaluate!(model, h)
            y[k] = h[1, output] - f0
            for p in 0:degree
                X[k, p+1] = xi^p
            end
        end
        β = X \ y
        f1[i] = (xi -> sum(β[p+1]*xi^p for p=0:degree))
    end

    # Second order
    f2 = Dict{Tuple{Int,Int}, Any}()
    if order >= 2
        for i in 1:d-1, j in i+1:d
            pts = [(xi, xj) for xi in xs, xj in xs]
            X = zeros(length(pts), (degree+1)^2)
            y = zeros(length(pts))
            for (k, (xi, xj)) in enumerate(pts)
                h = deepcopy(anchor)
                h[!, varnames[i]] .= xi
                h[!, varnames[j]] .= xj
                evaluate!(model, h)
                y[k] = h[1, output] - f1[i](xi) - f1[j](xj) - f0
                col = 1
                for pi in 0:degree, pj in 0:degree
                    X[k, col] = xi^pi * xj^pj
                    col += 1
                end
            end
            β = X \ y
            f2[(i, j)] = ((xi, xj) -> begin
                col = 1
                s = 0.0
                for pi in 0:degree, pj in 0:degree
                    s += β[col]*xi^pi*xj^pj
                    col += 1
                end
                return s
            end)
        end
    end

    return HDMRRepresentation(anchor_vec, f0, f1, f2, inputs, output)
end