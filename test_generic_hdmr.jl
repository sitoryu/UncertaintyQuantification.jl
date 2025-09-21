# Test Generic HDMR Implementation

using Pkg
Pkg.activate(".")
using UncertaintyQuantification
using DataFrames
using Distributions

f(df) = df.x1 .+ df.x2 .+ df.x3 .+ df.x1 .* df.x2 .+ 0.5 .* df.x1 .* df.x2 .* df.x3

inputs = [
    RandomVariable(Normal(0,2), :x1), 
    RandomVariable(Uniform(0,2), :x2),
    RandomVariable(Uniform(0,2), :x3)
]
model = Model(f, :y)
anchor = DataFrame(x1=[0.5], x2=[0.5], x3=[0.5])

println("Testing Generic HDMR with 3rd order...")

hdmr = cut_HDMR(model, inputs, :y, anchor; order=1, degree=2, samples=10)

for test_order in [1, 2, 3]
    println("\n=== Testing Order $test_order ===")
    
    try
        hdmr = cut_HDMR(model, inputs, :y, anchor; order=test_order, degree=2, samples=10)
        
        test_points = [
            [0.2, 0.3, 0.4],
            [0.8, 0.1, 0.9],
            [0.0, 1.0, 0.5]
        ]
        
        for point in test_points
            x1, x2, x3 = point
            hdmr_val = hdmr(point)
            true_val = x1 + x2 + x3 + x1*x2 + 0.5*x1*x2*x3
            error = abs(hdmr_val - true_val)
            
            println("Point $point:")
            println("  HDMR: $hdmr_val")
            println("  True: $true_val") 
            println("  Error: $error")
        end
        
    catch e
        println("Error at order $test_order: $e")
        rethrow(e)
    end
end
