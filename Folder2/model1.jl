"""
Initial model
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions
using JuMP, Ipopt
using CSV, DataFrames
using Plots

# generate series
Random.seed!(123)
n = 300
p = 2
s = 12

Y = Matrix{Float64}(undef, n, p)
v = 3
Σ = [1 0.5; 0.5 1]
ϕ = [0.5, 0.5]
mu_til = [70, 80]
m = 75
g = sin.(range(0, 2π, length=s))
g = 1 * (g .- mean(g))
for t = 1:n
    m = (1 .- ϕ) .* mu_til + ϕ .* m
    gam = g[t%12+1]
    mu = m .+ gam
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
model = Model(Ipopt.Optimizer)
set_attribute(model, "max_cpu_time", 60.0)
# declare θ: decision variables
@variables(model, begin
    m[1:n, 1:p]
    μ[1:n, 1:p]
    S[1:n, 1:p]
    ω[1:p]
    ϕ[1:p]
    k[1:1, 1:p]
    m₀[1:p]
end);
# declare constraints: S and μ (d=0)
@constraint(model, [t = 1:n, i = 1:p], S[t, i] == (
    -(v + p) *
    inv(v) *
    (inv(Σ)*(Y[t, :]-μ[t, :]))[i] *
    inv(
        1 +
        inv(v) *
        (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :])
    )
));
@constraints(model, begin
    [i = 1:p], m[1, i] == (1 - ϕ[i]) * ω[i] + ϕ[i] * m₀[i]
    [t = 2:n, i = 1:p], m[t, i] == (1 - ϕ[i]) * ω[i] + ϕ[i] * m[t-1, i] + k[1, i] * S[t-1, i]
    [t = 1:n, i = 1:p], μ[t, i] == m[t, i]
end);
# bound constraints
@constraints(model, begin
    k .≤ 5
    k .≥ -5
    ϕ .≤ 1
    ϕ .≥ -1
end);
# log loss function as objective
pre_logL = log(gamma((v + p) / 2)) - log(gamma(v / 2)) - 0.5 * p * log(v * π) - 0.5 * log(det(Σ));
LogL = @expression(model, [t = 1:n], pre_logL - (v + p) * 0.5 * log(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :])));
@objective(model, Max, sum(LogL[1:end]));
# set starting values
set_start_value.(μ[1:end, :], Y[1:end, :]);
# solve the optimization problem
optimize!(model)

plot(Y[:, 1], label="y₁", color="black")
plot!(value.(μ[:, 1]), label="μ₁")
plot!(value.(m[:, 1]), label="m₁")

plot(Y[:, 2], label="y₂", color="black")
plot!(value.(μ[:, 2]), label="μ₂")
plot!(value.(m[:, 2]), label="m₂")

println("ϕ values:")
value.(ϕ)
println("ω values:")
value.(ω)
println("m₀ values:")
value.(m₀)
println("k values:")
value.(k)

plot(value.(Y[:, 1] - μ[:, 1]), label="ϵ₁")
plot!(value.(Y[:, 2] - μ[:, 2]), label="ϵ₂")

plot(value.(S[:, 1]), label="S₁")
plot!(value.(S[:, 2]), label="S₂")

M = Matrix{Float64}(undef, n, 2)
for t = 1:n
    M[t, 1] = value(LogL[t])
    M[t, 2] = log(pdf(MvTDist(v, value.(μ)[t, :], Σ), Y[t, :]))
end
maximum(abs.(M[:, 1] - M[:, 2]))