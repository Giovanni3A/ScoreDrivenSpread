"""
prediction for the spread series using the fitted model.
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions, StatsBase
using Optim
using CSV, DataFrames
using Plots
using StatsPlots
using ProgressBars

include("loglikelihood.jl")

df = CSV.read("projeto//ScoreDrivenSpread//data//trusted//monthly_data.csv", DataFrame)
fit_df = CSV.read("projeto//ScoreDrivenSpread//data//results//model1//μ_hat.csv", DataFrame; delim=";", decimal=',')
params_df = CSV.read("projeto//ScoreDrivenSpread//data//results//model1//params.csv", DataFrame; delim=";", decimal=',')

# get full size
orig_Y = Array(df[:, 2:5])
Y = copy(orig_Y)
n, p = size(Y)


# boolean to 1 month ahead prediction
h1 = false

# load fit params
m₀ = params_df[1:4, 2]
ω = params_df[5:8, 2]
ϕ = params_df[9:12, 2]
κ_trend = params_df[13:16, 2]
κ_sazon = params_df[17:20, 2]
Σ = params_df[21:30, 2]
v = params_df[31, 2]
γ = reshape(params_df[32:end, 2], 11, 4)
γ₀₁ = γ[:, 1]
γ₀₂ = γ[:, 2]
γ₀₃ = γ[:, 3]
γ₀₄ = γ[:, 4]

# mount seasonality dummies matrix
D = zeros(n, 12)
for t = 1:n
    s_t = t % 12 + 1
    D[t, s_t] = 1
end

# apply transformations
ϕ = sigmoid.(ϕ)
m₀ = sigmoid.(m₀)
γ₀₁ = [γ₀₁; -sum(γ₀₁)]
γ₀₂ = [γ₀₂; -sum(γ₀₂)]
γ₀₃ = [γ₀₃; -sum(γ₀₃)]
γ₀₄ = [γ₀₄; -sum(γ₀₄)]
chol_inv_Σ = [
    Σ[1] Σ[2] Σ[3] Σ[4];
    0.0 Σ[5] Σ[6] Σ[7];
    0.0 0.0 Σ[8] Σ[9];
    0.0 0.0 0.0 Σ[10]
]
inv_Σ = chol_inv_Σ * chol_inv_Σ'
v = exp(v) + 2

# initialize variables
m = Matrix{Float64}(undef, n, p)
γ = zeros(n, 12, p)
μ = Matrix{Float64}(undef, n, p)
S = Matrix{Float64}(undef, n, p)

# loop over (train) time
for t = 1:n-48

    # trend component
    if t == 1
        m[t, :] = (1 .- ϕ) .* ω .+ ϕ .* m₀
    else
        m[t, :] = (1 .- ϕ) .* ω + ϕ .* m[t-1, :] + κ_trend .* S[t-1, :]
    end

    # seasonality component
    aₜ = (-1 / 11) * (ones(12) - 12 * D[t, :])
    if t == 1
        γ[t, :, 1] = γ₀₁
        γ[t, :, 2] = γ₀₂
        γ[t, :, 3] = γ₀₃
        γ[t, :, 4] = γ₀₄
    else
        γ[t, :, 1] = γ[t-1, :, 1] + aₜ * κ_sazon[1] * S[t-1, 1]
        γ[t, :, 2] = γ[t-1, :, 2] + aₜ * κ_sazon[2] * S[t-1, 2]
        γ[t, :, 3] = γ[t-1, :, 3] + aₜ * κ_sazon[3] * S[t-1, 3]
        γ[t, :, 4] = γ[t-1, :, 4] + aₜ * κ_sazon[4] * S[t-1, 4]
    end

    # μₜ
    μ[t, :] = m[t, :] + γ[t, :, :]'D[t, :]

    # Sₜ (score)
    S[t, :] = (
        (v + p) *
        inv(v) *
        (inv_Σ * (Y[t, :] - μ[t, :])) *
        inv(
            1 +
            inv(v) *
            (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :])
        )
    )

end

J = 30000
future_Y = []
for j in ProgressBar(1:J)
    Y1 = copy(Y)
    for t = n-48+1:n

        m[t, :] = (1 .- ϕ) .* ω + ϕ .* m[t-1, :] + κ_trend .* S[t-1, :]
        aₜ = (-1 / 11) * (ones(12) - 12 * D[t, :])
        γ[t, :, 1] = γ[t-1, :, 1] + aₜ * κ_sazon[1] * S[t-1, 1]
        γ[t, :, 2] = γ[t-1, :, 2] + aₜ * κ_sazon[2] * S[t-1, 2]
        γ[t, :, 3] = γ[t-1, :, 3] + aₜ * κ_sazon[3] * S[t-1, 3]
        γ[t, :, 4] = γ[t-1, :, 4] + aₜ * κ_sazon[4] * S[t-1, 4]
        μ[t, :] = m[t, :] + γ[t, :, :]'D[t, :]

        dist = MvTDist(exp(v) + 2, μ[t, :], round.(inv(inv_Σ), digits=6))
        if h1
            Y1[t, :] = rand(dist)
        else
            Y[t, :] = rand(dist)
        end

        S[t, :] = (
            (v + p) *
            inv(v) *
            (inv_Σ * (Y[t, :] - μ[t, :])) *
            inv(
                1 +
                inv(v) *
                (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :])
            )
        )
    end
    if h1
        append!(future_Y, [copy(Y1)])
    else
        append!(future_Y, [copy(Y1)])
    end
end

plot_range = n-200:n
for i = 1:p
    println(i)
    fig = plot(df[plot_range, 1], orig_Y[plot_range, i], color="black", label="Spread Series")
    l = []
    u = []
    for t = n-48:n
        append!(l, quantile([future_Y[j][t, i] for j = 1:J], 0.05))
        append!(u, quantile([future_Y[j][t, i] for j = 1:J], 0.95))
    end
    plot!(df[n-48:n, 1], (l .+ u) ./ 2, ribbon=(l .- u) ./ 2, fillalpha=0.35, c=1, label="95% Confidence band")
    if h1
        savefig(fig, "projeto//ScoreDrivenSpread//data//results//model1//forecast_1ahead_$i.png")
    else
        savefig(fig, "projeto//ScoreDrivenSpread//data//results//model1//forecast_$i.png")
    end
end

# create dataframe with lower and upper bounds
predict_df = DataFrame()
predict_df[!, Symbol("date")] = df[n-48:n, 1]
for i=1:p
    l = []
    u = []
    a = []
    for t = n-48:n
        append!(l, quantile([future_Y[j][t, i] for j = 1:J], 0.05))
        append!(u, quantile([future_Y[j][t, i] for j = 1:J], 0.95))
        append!(a, mean([future_Y[j][t, i] for j = 1:J]))
    end
    predict_df[!, Symbol("l$i")] = l
    predict_df[!, Symbol("y$i")] = a
    predict_df[!, Symbol("u$i")] = u
end

predict_df

CSV.write("projeto//ScoreDrivenSpread//data//results//model1//forecast.csv", predict_df; delim=";", decimal=',')
