using Distributions, Random
using LinearAlgebra, SpecialFunctions
using JuMP, Ipopt
using CSV, DataFrames
using Plots

# generate series
Random.seed!(370)
n = 300
p = 2
Y = Matrix{Float64}(undef, n, p)
v = 3
Σ = Matrix(I(p))
ϕ = [0.5, 0.5]
mu_til = [3, 5]
m = 0
for t = 1:n
    m = (1 .- ϕ) .* mu_til + ϕ .* m
    mu = m
    dist = MvTDist(
        v,
        mu,
        Σ
    )
    Y[t, :] = rand(dist)
end

plot(Y[:, 1], label="y₁", title="Original Series")
plot!(Y[:, 2], label="y₂")

# create the optimization model
m = Model(Ipopt.Optimizer)
set_attribute(m, "max_cpu_time", 60.0)
# declare θ: decision variables
@variables(m, begin
    μ[1:n, 1:p]
    S[1:n, 1:p]
    ω[1:p]
    ϕ[1:p]
    k[1:p]
    μ₀[1:p]
end)
# declare constraints: S and μ (d=0)
@constraint(m, [t = 1:n, i = 1:p], S[t, i] == (
    -(v + p) * inv(v) * (inv(Σ)*(Y[t, :]-μ[t, :]))[i] * inv(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :]))
))
@constraints(m, begin
    [i = 1:p], μ[1, i] == (1 - ϕ[i]) * ω[i] + ϕ[i] * μ₀[i] + k[i] * S[1, i]
    [t = 2:n, i = 1:p], μ[t, i] == (1 - ϕ[i]) * ω[i] + ϕ[i] * μ[t-1, i] + k[i] * S[t, i]
end)
# bound constraints
@constraints(m, begin
    k .≤ 30
    k .≥ -30
    # ϕ .≤ 1
    # ϕ .≥ -1
    # ω .≥ -20
    # ω .≤ 20
    # μ₀ .≥ -20
    # μ₀ .≤ 20
    # μ .≥ -10
    # μ .≤ 20
    # S .≥ -10
    # S .≤ 10
end)
# log loss function as objective
pre_logL = log(gamma((v + p) / 2)) - log(gamma(v / 2)) - 0.5 * p * log(v * π) - 0.5 * log(det(Σ))
LogL = @expression(m, [t = 1:n], pre_logL - (v + p) * 0.5 * log(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :])))
@objective(m, Max, sum(LogL[1:end]))
# set starting values
set_start_value.(μ[1:end, :], Y[1:end, :])
# solve the optimization problem
optimize!(m)

plot(Y[:, 1], label="y₁")
plot!(value.(μ[:, 1]), label="μ₁")

plot(value.(Y[:, 2]), label="y₂")
plot!(value.(μ[:, 2]), label="μ₂")

println("ϕ values:")
value.(ϕ)
println("ω values:")
value.(ω)
println("μ₀ values:")
value.(μ₀)
println("k values:")
value.(k)

plot(value.(Y[:, 1] - μ[:, 1]), label="ϵ₁")
plot!(value.(Y[:, 2] - μ[:, 2]), label="ϵ₂")

plot(value.(S[:, 1]), label="S₁")
plot(value.(S[:, 2]), label="S₂")

M = Matrix{Float64}(undef, n, 2)
for t = 1:n
    M[t, 1] = value(LogL[t])
    M[t, 2] = log(pdf(MvTDist(v, value.(μ)[t, :], Σ), Y[t, :]))
end
maximum(abs.(M[:, 1] - M[:, 2]))