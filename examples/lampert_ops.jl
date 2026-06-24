using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "src", "lib", "julia"))

using JSON
using LinearAlgebra
using ForwardDiff
using Manifolds
using Manopt
using SummationByPartsOperators
using SummationByPartsOperatorsExtra
import ADTypes
import Optim


alpha = 2.75
N = 15

#basis = [one, identity, x -> x^2, x -> x^3]
basis = [x -> 1.0, x -> x, x -> x^2, x -> x^3, x -> exp(alpha * x)]
#reg_funcs = [x -> x^4]
nodes = collect(LinRange(0.0, 1.0, N))

#source_opt = GlaubitzIskeLampertÖffner2026Regularized()

println("\n")
println("Basic operator:")

# The Lampert sources use the Manopt backend, so options are Manopt keyword
# arguments, not an Optim.Options object. Optim.Options is valid for the older
# Optim-backed sources such as GlaubitzNordströmÖffner2023.
stopping_criterion = StopAfterIteration(10000) |
    StopWhenGradientNormLess(1e-16) |
    StopWhenCostLess(1e-28)
debug = [:Iteration, :Time, " | ", (:Cost, "f(x): %.6e"), " | ",
     (:GradientNorm, "||∇f(x)||: %.6e"),
     "\n", :Stop]
basic_options = (;
    SummationByPartsOperatorsExtra.default_options(source_basic, true)...,
    stopping_criterion = stopping_criterion,
    debug = debug,
)

D_basic = SummationByPartsOperatorsExtra.function_space_operator(
    basis,
    nodes,
    GlaubitzIskeLampertÖffner2026Basic();
    autodiff = ADTypes.AutoForwardDiff(),
    verbose = true,
    options = basic_options
)

println("\n")
println("Basic operator 2:")

D_basic2 = SummationByPartsOperatorsExtra.function_space_operator(
    basis,
    nodes,
    GlaubitzNordströmÖffner2023();
    autodiff = ADTypes.AutoForwardDiff(),
    verbose = true,
    options=Optim.Options(g_tol=1e-19, iterations=10000, show_trace=false)
)

# println("\n")
# println("Optimized operator:")

# x0 = get_optimization_entries(D_basic)
# stopping_criterion = StopAfterIteration(4) |
#     #  StopWhenGradientNormLess(1e-16) |
#     StopWhenCostLess(1e-28)
# D_opt = SummationByPartsOperatorsExtra.function_space_operator(
#     basis,
#     nodes,
#     source_opt;
#     x0=x0,
#     regularization_functions = reg_funcs,
#     #autodiff = ADTypes.AutoForwardDiff(),
#     verbose = true,
#     options=(;
#         debug=debug,
#         stopping_criterion=stopping_criterion,
#     )
# )


#D_mat = Matrix(D_opt)
#H_vec = diag(mass_matrix(D_opt))
#tL_vec = SummationByPartsOperators.left_boundary_weight(D_opt)
#tR_vec = SummationByPartsOperators.right_boundary_weight(D_opt)

println("\nTesting:")
#println("rank(D_basic) = $(rank(Matrix(D_basic))), rank(D_opt) = $(rank(Matrix(D_opt)))")
println("rank(D) = $(rank(Matrix(D_basic)))")

println("\nBasic operator:")
D_mat = Matrix(D_basic)
P = mass_matrix(D_basic)
B = mass_matrix_boundary(D_basic)
sbp_residual = P * D_mat + D_mat' * P - B
println("SBP residual max |error| = ", maximum(abs.(sbp_residual)))

test_functions = [x -> x^(i - 1) for i in 1:5]
test_functions_derivatives = [zero, (x -> (i - 1) * x^(i - 2) for i in 2:N)...]
println("Derivative residuals:")
for (i, (f, f_deriv)) in enumerate(zip(test_functions, test_functions_derivatives))
    deriv_residual = D_mat * f.(nodes) - f_deriv.(nodes)
    println("  test $i: max |error| = ", maximum(abs.(deriv_residual)))
end

println("\nBasic operator 2:")
D_mat = Matrix(D_basic2)
P = mass_matrix(D_basic2)
B = mass_matrix_boundary(D_basic2)
sbp_residual = P * D_mat + D_mat' * P - B
println("SBP residual max |error| = ", maximum(abs.(sbp_residual)))

test_functions = [x -> x^(i - 1) for i in 1:5]
test_functions_derivatives = [zero, (x -> (i - 1) * x^(i - 2) for i in 2:N)...]
println("Derivative residuals:")
for (i, (f, f_deriv)) in enumerate(zip(test_functions, test_functions_derivatives))
    deriv_residual = D_mat * f.(nodes) - f_deriv.(nodes)
    println("  test $i: max |error| = ", maximum(abs.(deriv_residual)))
end


# println("\nOptimized operator:")
# D_mat = Matrix(D_opt)
# P = mass_matrix(D_opt)
# B = mass_matrix_boundary(D_opt)
# sbp_residual = P * D_mat + D_mat' * P - B
# println("SBP residual max |error| = ", maximum(abs.(sbp_residual)))

# test_functions = [x -> x^(i - 1) for i in 1:5]
# test_functions_derivatives = [zero, (x -> (i - 1) * x^(i - 2) for i in 2:N)...]
# println("Derivative residuals:")
# for (i, (f, f_deriv)) in enumerate(zip(test_functions, test_functions_derivatives))
#     deriv_residual = D_mat * f.(nodes) - f_deriv.(nodes)
#     println("  test $i: max |error| = ", maximum(abs.(deriv_residual)))
# end
