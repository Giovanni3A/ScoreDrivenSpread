using Distributions, Random
using LinearAlgebra, SpecialFunctions
using JuMP, Ipopt
using CSV, DataFrames
using Plots

# read data from monthly_data.csv file
df = CSV.read("projeto//ScoreDrivenSpread//dados//monthly_data.csv", DataFrame)
y1 = df[:, 2]
y2 = df[:, 3]
y3 = df[:, 4]
y4 = df[:, 5]
Y = [y1 y2 y3 y4][:, 1:3]

plot(Y[:, 1], label="y₁", title="Original Series")
plot!(Y[:, 2], label="y₂")
plot!(Y[:, 3], label="y₃")
plot!(Y[:, 4], label="y₄")

n, p = size(Y)
v = 3
Σ = Matrix(I(p))

# create the optimization model
model = Model(Ipopt.Optimizer)
set_attribute(model, "max_cpu_time", 60.0)
# declare θ: decision variables
@variables(model, begin
    m[1:n, 1:p]
    γ[1:n, 1:p]
    μ[1:n, 1:p]
    S[1:n, 1:p]
    ω[1:p]
    ϕ[1:p]
    k[1:2, 1:p]
    m₀[1:p]
    γ₀[1:11, 1:p]
end)
# declare constraints: S and μ (d=0)
@constraint(model, [t = 1:n, i = 1:p], S[t, i] == (
    -(v + p) * inv(v) * (inv(Σ)*(Y[t, :]-μ[t, :]))[i] * inv(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :]))
));
@constraints(model, begin
    [i = 1:p], m[1, i] == (1 - ϕ[i]) * ω[i] + ϕ[i] * m₀[i] + k[i] * S[1, i]
    [t = 2:n, i = 1:p], m[t, i] == (1 - ϕ[i]) * ω[i] + ϕ[i] * m[t-1, i] + k[1, i] * S[t, i]
    [t = 12:n, i = 1:p], γ[t, i] == -sum(γ[t-11:t-1, i]) + k[2, i] * S[t, i]
    [t = 1:n, i = 1:p], μ[t, i] == m[t, i] + γ[t, i]
end);
for t = 1:11
    @constraint(model, [i = 1:p], γ[t, i] == -(sum(γ₀[t:end, i])) + sum(γ[1:t-1, i]))
end
# bound constraints
@constraints(model, begin
    k .≤ 50
    k .≥ -50
    ϕ .≤ 1
    ϕ .≥ -1
end)
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
plot!(value.(γ[:, 1]), label="γ₁")

plot(Y[:, 2], label="y₂", color="black")
plot!(value.(μ[:, 2]), label="μ₂")
plot!(value.(m[:, 2]), label="m₂")
plot!(value.(γ[:, 2]), label="γ₂")

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
plot!(value.(Y[:, 3] - μ[:, 3]), label="ϵ₃")

plot(value.(S[:, 1]), label="S₁")
plot!(value.(S[:, 2]), label="S₂")
plot!(value.(S[:, 3]), label="S₃")

M = Matrix{Float64}(undef, n, 2)
for t = 1:n
    M[t, 1] = value(LogL[t])
    M[t, 2] = log(pdf(MvTDist(v, value.(μ)[t, :], Σ), Y[t, :]))
end
maximum(abs.(M[:, 1] - M[:, 2]))