using Distributions, Random
using LinearAlgebra, SpecialFunctions
using JuMP, Ipopt
using CSV, DataFrames
using Plots

# TODO: completar a função objetivo
# TODO: ponto inicial
# TODO: comparar com multi-t implementada no julia

# generate series
Random.seed!(370)
n = 200
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
set_attribute(m, "warm_start_init_point", "no")
# declare θ: decision variables
@variables(m, begin
    μ[1:n, 1:p]
    S[1:n, 1:p]
    ω[1:p]
    ϕ[1:p]
    k[1:p]
end)
# declare constraints: S and μ (d=0)
@constraint(m, [t = 1:n, i = 1:p], S[t, i] == (
    -(v + p) * inv(v) * (inv(Σ)*(Y[t, :]-μ[t, :]))[i] * inv(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :]))
))
@constraints(m, begin
    [t = 2:n, i = 1:p], μ[t, i] == (1 - ϕ[i]) * ω[i] + ϕ[i] * μ[t-1, i] + k[i] * S[t, i]
end)
# bound constraints
@constraints(m, begin
    k .≤ 10
    k .≥ -10
end)
# log loss function as objective
pre_logL = log(gamma((v + p) / 2)) - log(gamma(v / 2)) - 0.5 * p * log(v * π) - 0.5 * log(det(Σ))
LogL = @expression(m, [t = 1:n], pre_logL - (v + p) * 0.5 * log(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :])))
@objective(m, Max, sum(LogL[1:end]))
# optimize
optimize!(m)

plot(Y[:, 1], label="y₁")
plot!(value.(μ[:, 1]), label="μ₁")

plot(value.(Y[:, 2]), label="y₂")
plot!(value.(μ[:, 2]), label="μ₂")

println("ϕ values:")
value.(ϕ)
println("ω values:")
value.(ω)

plot(value.(Y[:, 1] - μ[:, 1]), label="ϵ₁")
plot!(value.(Y[:, 2] - μ[:, 2]), label="ϵ₂")

value.(k)

M = Matrix{Float64}(undef, n, 2)
for t = 1:n
    M[t, 1] = value(LogL[t])
    M[t, 2] = log(pdf(MvTDist(v, value.(μ)[t, :], Σ), Y[t, :]))
end
maximum(abs.(M[:, 1] - M[:, 2]))