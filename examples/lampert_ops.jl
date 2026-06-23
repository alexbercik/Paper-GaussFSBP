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

function export_exponential_operators(filename::String)
    cache = Dict{String, Any}()

    alpha = 4.0

    basis_3 = (x -> 1.0, x -> x, x -> exp(alpha * x))
    reg_funcs = (x -> x^2, x -> x^3)
    nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    source = GlaubitzIskeLampertÖffner2026Regularized()
    #source = GlaubitzIskeLampertÖffner2026Basic()

    D_op = SummationByPartsOperatorsExtra.function_space_operator(
        basis_3,
        nodes,
        source;
        regularization_functions = reg_funcs,
        autodiff = ADTypes.AutoForwardDiff(),
        verbose = true
    )

    D_mat = Matrix(D_op)
    H_vec = diag(mass_matrix(D_op))
    tL_vec = SummationByPartsOperators.left_boundary_weight(D_op)
    tR_vec = SummationByPartsOperators.right_boundary_weight(D_op)

    cache_key = "exp_k1_opt_p2_open_float64"
    cache[cache_key] = Dict(
        "k" => 1,
        "node_type" => "opt",
        "basis" => ["1", "x", "exp($alpha*x)"],
        "quad_basis" => ["1", "x", "x^2", "x^3", "exp($alpha*x)", "x*exp($alpha*x)"],
        "op_type" => "closed",
        "selector" => 0,
        "interval" => [0.0, 1.0],
        "nodes" => nodes,
        "D" => [row for row in eachrow(D_mat)],
        "H" => H_vec,
        "tL" => tL_vec,
        "tR" => tR_vec
    )

    open(filename, "w") do f
        JSON.print(f, cache, 4)
    end
    
    println("Successfully exported operator cache to ", filename)
end

export_exponential_operators(joinpath(@__DIR__, "operator_cache_lampert.json"))
