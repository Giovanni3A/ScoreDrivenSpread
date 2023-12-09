"""
Apply real data
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions, StatsBase
using JuMP, Ipopt
using CSV, DataFrames
using Plots

# read data from monthly_data.csv file
df = CSV.read("projeto//ScoreDrivenSpread//dados//monthly_data.csv", DataFrame)
y1 = df[:, 7]
y2 = df[:, 8]
y3 = df[:, 9]
y4 = df[:, 10]
Y = [y1 y2 y3 y4][:, 1:4]

plot(Y[:, 1], label="y₁", title="Spread Series")
plot!(Y[:, 2], label="y₂")
plot!(Y[:, 3], label="y₃")
plot!(Y[:, 4], label="y₄")

n, p = size(Y)
v = 3
Σ = cov(Y) * (v - 2) / v

# mount seasonality dummies matrix
D = zeros(n, 12)
for t = 1:n
    s_t = t % 12 + 1
    D[t, s_t] = 1
end

# create the optimization model
model = Model(Ipopt.Optimizer)
set_attribute(model, "max_cpu_time", 60.0 * 10)
# declare θ: decision variables
@variables(model, begin
    m[1:n, 1:p]
    γ[1:n, 1:p, 1:12]
    μ[1:n, 1:p]
    S[1:n, 1:p]
    ω[1:p]
    ϕ[1:p]
    k[1:2, 1:p]
    m₀[1:p]
    γ₀[1:p, 1:12]
end);
# declare constraints: S and μ (d=0)
@constraint(model, [t = 1:n, i = 1:p], S[t, i] == (
    (v + p) *
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
    [t = 1:n, i = 1:p], μ[t, i] == m[t, i] + D[t, :]'γ[t, i, :]
end);
# HS seasonality
for t = 1:n
    j = t % 12 + 1
    for s = 1:12
        if s == j
            kappa = 1
        else
            kappa = -1 / 11
        end
        if t == 1
            @constraint(model, [i = 1:p], γ[t, i, s] == γ₀[i, s])
        else
            @constraint(model, [i = 1:p], γ[t, i, s] == γ[t-1, i, s] + kappa * k[2, i] * S[t-1, i])
        end
    end
end
@constraint(model, [i = 1:p], sum(γ₀[i, :]) == 0)
# bound constraints
@constraints(model, begin
    k .≤ 5
    k .≥ -5
    ϕ .≤ 1
    ϕ .≥ 0
    m₀ .≥ -1
    m₀ .≤ 1
    ω .≥ -1
    ω .≤ 1
end);
# log loss function as objective
pre_logL = log(gamma((v + p) / 2)) - log(gamma(v / 2)) - 0.5 * p * log(v * π) - 0.5 * log(det(Σ));
LogL = @expression(model, [t = 1:n], pre_logL - (v + p) * 0.5 * log(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :])));
@objective(model, Max, sum(LogL[1:end]));
# set starting values
set_start_value.(μ[1:end, :], Y[1:end, :]);
# solve the optimization problem
optimize!(model)

fig1 = plot(Y[:, 1], label="y₁", color="black")
plot!(value.(μ[:, 1]), label="μ₁")
plot!(value.(m[:, 1]), label="m₁")
savefig(fig1, "projeto//ScoreDrivenSpread//Folder2//pred1.png")
plot!(twinx(), [value(D[t, :]'γ[t, 1, :]) for t = 1:n], label="γ₁")

fig2 = plot(Y[:, 2], label="y₂", color="black")
plot!(value.(μ[:, 2]), label="μ₂")
plot!(value.(m[:, 2]), label="m₂")
savefig(fig2, "projeto//ScoreDrivenSpread//Folder2//pred2.png")
plot!(twinx(), [value(D[t, :]'γ[t, 2, :]) for t = 1:n], label="γ₂")

fig3 = plot(Y[:, 3], label="y₃", color="black")
plot!(value.(μ[:, 3]), label="μ₃")
plot!(value.(m[:, 3]), label="m₃")
savefig(fig3, "projeto//ScoreDrivenSpread//Folder2//pred3.png")
plot!(twinx(), [value(D[t, :]'γ[t, 3, :]) for t = 1:n], label="γ₃")

fig4 = plot(Y[:, 4], label="y₄", color="black")
plot!(value.(μ[:, 4]), label="μ₄")
plot!(value.(m[:, 4]), label="m₄")
savefig(fig4, "projeto//ScoreDrivenSpread//Folder2//pred4.png")
plot!(twinx(), [value(D[t, :]'γ[t, 4, :]) for t = 1:n], label="γ₄")

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
plot!(value.(Y[:, 3] - μ[:, 3]), label="ϵ₃")
plot!(value.(Y[:, 4] - μ[:, 4]), label="ϵ₄")

fig = bar(autocor(value.(Y[:, 1] - μ[:, 1]), 1:15), label="y₁", alpha=0.5)
bar!(autocor(value.(Y[:, 2] - μ[:, 2]), 1:15), label="y₂", alpha=0.5)
bar!(autocor(value.(Y[:, 3] - μ[:, 3]), 1:15), label="y₃", alpha=0.5)
bar!(autocor(value.(Y[:, 4] - μ[:, 4]), 1:15), label="y₄", alpha=0.5)
savefig(fig, "projeto//ScoreDrivenSpread//Folder2//autocor.png")

plot(value.(S[:, 1]), label="S₁")
plot!(value.(S[:, 2]), label="S₂")
plot!(value.(S[:, 3]), label="S₃")
plot!(value.(S[:, 4]), label="S₄")

M = Matrix{Float64}(undef, n, 2)
for t = 1:n
    M[t, 1] = value(LogL[t])
    M[t, 2] = log(pdf(MvTDist(v, value.(μ)[t, :], Σ), Y[t, :]))
end
maximum(abs.(M[:, 1] - M[:, 2]))

# save parameters csv file
params_df = DataFrame(
    name=[
        "m01", "m02", "m03", "m04",
        "omega1", "omega2", "omega3", "omega4",
        "phi1", "phi2", "phi3", "phi4",
        "kappa1", "kappa2", "kappa3", "kappa4", "kappa5", "kappa6", "kappa7", "kappa8",
        "gamma_1_1", "gamma_1_2", "gamma_1_3", "gamma_1_4", "gamma_1_5", "gamma_1_6", "gamma_1_7", "gamma_1_8", "gamma_1_9", "gamma_1_10", "gamma_1_11", "gamma_1_12",
        "gamma_2_1", "gamma_2_2", "gamma_2_3", "gamma_2_4", "gamma_2_5", "gamma_2_6", "gamma_2_7", "gamma_2_8", "gamma_2_9", "gamma_2_10", "gamma_2_11", "gamma_2_12",
        "gamma_3_1", "gamma_3_2", "gamma_3_3", "gamma_3_4", "gamma_3_5", "gamma_3_6", "gamma_3_7", "gamma_3_8", "gamma_3_9", "gamma_3_10", "gamma_3_11", "gamma_3_12",
        "gamma_4_1", "gamma_4_2", "gamma_4_3", "gamma_4_4", "gamma_4_5", "gamma_4_6", "gamma_4_7", "gamma_4_8", "gamma_4_9", "gamma_4_10", "gamma_4_11", "gamma_4_12",
    ],
    value=[
        value.(m₀)...,
        value.(ω)...,
        value.(ϕ)...,
        value.(k[1, :])..., value.(k[2, :])...,
        value.(γ₀')...
    ]
)
CSV.write("projeto//ScoreDrivenSpread//Folder2//params.csv", params_df; delim=";", decimal=',')

# save series csv file
series_df = DataFrame(
    t=1:n,
    true1=Y[:, 1],
    true2=Y[:, 2],
    true3=Y[:, 3],
    true4=Y[:, 4],
    prev1=value.(μ[:, 1]),
    prev2=value.(μ[:, 2]),
    prev3=value.(μ[:, 3]),
    prev4=value.(μ[:, 4])
)
CSV.write("projeto//ScoreDrivenSpread//Folder2//series.csv", series_df; delim=";", decimal=',')
