using Distributions, Random
using JuMP, Ipopt
using CSV, DataFrames
using Plots

# TODO: usr série artificial
# TODO: ponto inicial
# TODO: comparar com multi-t implementada no julia

# read monthly_data
df = CSV.read("monthly_data.csv", DataFrame)

# get series
y1 = df[:, :spread1]
y2 = df[:, :spread2]
Y = [y1 y2]

# get dimensions
n = size(y1, 1)
p = 2

# fix parameters
v = 3
Σ = [var(y1) 0; 0 var(y2)]

# create the optimization model
m = Model(Ipopt.Optimizer)
set_attribute(m, "max_cpu_time", 60.0)
set_attribute(m, "tol", 1e-9)
set_attribute(m, "warm_start_init_point", "yes")
# declare θ: decision variables
@variables(m, begin
    μ[1:n, 1:p]
    S[1:n, 1:p]
    μ₀[1:2, 1:p]
    ϕ[1:2, 1:p]
    ω[1:p]
    k[1:p]
end)
# declare constraints: S and μ (d=0)
@constraint(m, [t = 1:n, i = 1:p], S[t, i] == (
    ((v+p)*inv(v - 2)*inv(Σ)*(Y[t, :]-μ[t, :]))[i] * inv(1 + inv(v - 2) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :]))
))
@constraints(m, begin
    [i = 1:p], μ[1, i] == ω[i] + ϕ[1, i] * μ₀[2, i] + ϕ[2, i] * μ₀[1, i] + k[i] * S[1, i]
    [i = 1:p], μ[2, i] == ω[i] + ϕ[1, i] * μ[1, i] + ϕ[2, i] * μ₀[2, i] + k[i] * S[2, i]
    [t = 3:n, i = 1:p], μ[t, i] == ω[i] + ϕ[1, i] * μ[t-1, i] + ϕ[2, i] * μ[t-2, i] + k[i] * S[t, i]
end)
# bound constraints
@constraints(m, begin
    k .≤ 2
    k .≥ -2
    ϕ .≤ 1 - 1e-3
    ϕ .≥ -(1 - 1e-3)
    ω .== 0
    ϕ[2, :] .== 0
end)
# log loss function as objective
LogL = @expression(m, [t = 1:n], -(v + p) * 0.5 * log(1 + inv(v - 2) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :])))
@objective(m, Max, sum(LogL))
# optimize
optimize!(m)

plot(value.(Y[:, 1]), label="y₁")
plot!(value.(μ[:, 1]), label="μ₁")
plot!(ones(n) * value(ω[1]), label="ω₁")

plot(value.(Y[:, 2]), label="y₂")
plot!(value.(μ[2:n, 2]), label="μ₂")
plot!(ones(n) * value(ω[2]), label="ω₂")

plot(value.(Y[:, 1] - μ[:, 1]), label="ϵ₁")
plot!(value.(Y[:, 2] - μ[:, 2]), label="ϵ₂")

value.(ϕ)
value.(k)
plot(value.(S))

