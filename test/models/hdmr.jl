@testset "Generic HDMR Implementation" begin

    f(df) = df.x1 .+ df.x2 .+ df.x3 .+ df.x1 .* df.x2 .+ 0.5 .* df.x1 .* df.x2 .* df.x3

    inputs = [
        RandomVariable(Normal(0,2), :x1), 
        RandomVariable(Uniform(0,2), :x2),
        RandomVariable(Uniform(0,2), :x3)
    ]
    model = Model(f, :y)
    anchor = DataFrame(x1=[0.5], x2=[0.5], x3=[0.5])

    test_points = [
        [0.2, 0.3, 0.4],
        [0.8, 0.1, 0.9],
        [0.0, 1.0, 0.5]
    ]

    for test_order in [1, 2, 3]
        @testset "Order $test_order" begin
            hdmr = cut_hdmr(model, inputs, :y, anchor; order=test_order, degree=3, samples=10)
            @test hdmr isa UncertaintyQuantification.HDMRRepresentation
            for point in test_points
                x1, x2, x3 = point
                hdmr_val = hdmr(point)
                true_val = x1 + x2 + x3 + x1*x2 + 0.5*x1*x2*x3
                error = abs(hdmr_val - true_val)
                @test isfinite(hdmr_val)
                tol = test_order == 1 ? 0.5 : test_order == 2 ? 0.05 : 0.02
                @test isapprox(hdmr_val, true_val; atol=tol)
            end
        end
    end

    @testset "_compute_lower_order_contribution" begin
        coeffs_dict = Dict([1] => [1.0, 2.0], [2] => [0.5, 1.5])
        indices = [1,2]
        base_value = 3.0
        args = [0.5, 0.2]
        degree = 1
        contrib = UncertaintyQuantification._compute_lower_order_contribution(coeffs_dict, indices, base_value, args, degree)
        @test isfinite(contrib)
        @test contrib >= base_value
    end
    @testset "_evaluate_polynomial_from_coeffs" begin
        coeffs = [1.0, 2.0, 3.0]
        args = [2.0]
        degree = 2
        val = UncertaintyQuantification._evaluate_polynomial_from_coeffs(coeffs, args, degree)
        # Should be 1*2^0 + 2*2^1 + 3*2^2 = 1 + 4 + 12 = 17
        @test isapprox(val, 17.0; atol=1e-10)
    end

    @testset "_build_polynomial_basis" begin
        sample_points = [[1.0], [2.0]]
        degree = 2
        num_vars = 1
        X = UncertaintyQuantification._build_polynomial_basis(sample_points, degree, num_vars)
        # For degree 2, 1 var: columns are [x^0, x^1, x^2]
        @test size(X) == (2, 3)
        @test X[1, :] == [1.0, 1.0, 1.0]
        @test X[2, :] == [1.0, 2.0, 4.0]
    end

    @testset "_combinations" begin
        arr = [1,2,3]
        k = 2
        combos = UncertaintyQuantification._combinations(arr, k)
        @test combos == [[1,2],[1,3],[2,3]]
        @test UncertaintyQuantification._combinations(arr, 0) == [[]]
        @test UncertaintyQuantification._combinations(arr, 3) == [[1,2,3]]
        @test UncertaintyQuantification._combinations(arr, 4) == []
    end

    @testset "HDMRRepresentation struct" begin
        hdmr = cut_hdmr(model, inputs, :y, anchor; order=1, degree=1, samples=5)
        @test hdmr.anchor isa Vector{Float64}
        @test hdmr.f0 isa Float64
        @test hdmr.coefficients isa Dict{Vector{Int}, Vector{Float64}}
        @test hdmr.degree isa Int
        @test hdmr.inputs isa Vector{UncertaintyQuantification.RandomVariable}
        @test hdmr.output == :y
        @test hdmr.max_order == 1
    end
end
