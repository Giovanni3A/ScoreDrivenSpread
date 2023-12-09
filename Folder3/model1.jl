"""
Initial model using Optim.jl
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions
using Optim
using CSV, DataFrames
using Plots

# read data from monthly_data.csv file
df = CSV.read("projeto//ScoreDrivenSpread//dados//monthly_data.csv", DataFrame)
y1 = df[:, 2]
y2 = df[:, 3]
y3 = df[:, 4]
y4 = df[:, 5]
Y = [y1 y2 y3 y4][:, 1:4]

plot(Y[:, 1], label="y₁", title="Original Series")
plot!(Y[:, 2], label="y₂")
plot!(Y[:, 3], label="y₃")
plot!(Y[:, 4], label="y₄")

# get size
n, p = size(Y)
Σ = cov(Y)
v = 3

# loss function
function loglikelihood(parameters)
    μ₀₁, μ₀₂, μ₀₃, μ₀₄, κ = parameters
    total = 0
    for t = 1:n
        μ = [μ₀₁, μ₀₂, μ₀₃, μ₀₄]
        dist = MvTDist(
            v,
            μ,
            Σ
        )
        total += log(pdf(dist, Y[t, :]))
    end
    return -total
end

initial = [Y[1, 1], Y[1, 2], Y[1, 3], Y[1, 4], 0.5]
lower_bound = [0, 0, 0, 0, -5]
upper_bound = [1000, 1000, 1000, 1000, 5]
res = optimize(loglikelihood, initial, NelderMead())
solution = Optim.minimizer(res)
μ₀₁, μ₀₂, μ₀₃, μ₀₄, κ = solution

plot(Y[:, 1], label="y₁", color="black")
plot!(fill(μ₀₁, n), label="μ₀₁")

plot(Y[:, 2], label="y₂", color="black")
plot!(fill(μ₀₂, n), label="μ₀₂")

println("κ value:")
κ
