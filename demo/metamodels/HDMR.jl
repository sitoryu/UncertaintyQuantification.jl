# cutHDMR example/test

using Pkg
Pkg.activate(".")
using UncertaintyQuantification
using DataFrames
using Distributions

f(df) = df.x1.^2 .+ 2 .* df.x2 .+ 1

inputs = [RandomVariable(Uniform(0,1), :x1), RandomVariable(Uniform(0,1), :x2)]
model = Model(f, :y)
anchor = DataFrame(x1=[0.5], x2=[0.5])

hdmr = cut_HDMR(model, inputs, :y, anchor; order=2, degree=2, samples=20)

println("HDMR at [0.2, 0.8]: ", hdmr([0.2, 0.8]))
println("True value at [0.2, 0.8]: ", 0.2^2 + 2*0.8 + 1)

println("HDMR at [1.0, 0.0]: ", hdmr([1.0, 0.0]))
println("True value at [1.0, 0.0]: ", 1.0^2 + 2*0.0 + 1)

println("HDMR at [0.0, 1.0]: ", hdmr([0.0, 1.0]))
println("True value at [0.0, 1.0]: ", 0.0^2 + 2*1.0 + 1)

errors = Float64[]
for x1 in [0.1, 0.3, 0.7, 0.9], x2 in [0.1, 0.3, 0.7, 0.9]
    hdmr_val = hdmr([x1, x2])
    true_val = x1^2 + 2*x2 + 1
    error = abs(hdmr_val - true_val)
    push!(errors, error)
    println("Point [$x1, $x2]: HDMR = $hdmr_val, True = $true_val, Error = $error")
end

println("\nMax error: ", maximum(errors))
println("Mean error: ", sum(errors) / length(errors))