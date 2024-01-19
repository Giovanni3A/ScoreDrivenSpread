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
fit_df = CSV.read("projeto//ScoreDrivenSpread//data//results//model2//μ_hat.csv", DataFrame; delim=";", decimal=',')
params_df = CSV.read("projeto//ScoreDrivenSpread//data//results//model2//params.csv", DataFrame; delim=";", decimal=',')

# future prediction window
w = 10*12

# get full size
orig_Y = Array(df[:, 2:5])
Y = copy(orig_Y)
n, p = size(Y)
Y = cat(Y, zeros(w, 4), dims=1)

# load fit params
m₀ = params_df[1:4, 2]
ω = params_df[5:8, 2]
ϕ = params_df[9:12, 2]
ψ₀ = params_df[13:16, 2]
ρ = params_df[17:20, 2]
η = params_df[21:24, 2]
κ_trend = params_df[25:28, 2]
κ_sazon = params_df[29:32, 2]
κ_var = params_df[33:36, 2]
Q = params_df[37:42, 2]
v = params_df[43, 2]
γ = reshape(params_df[44:end, 2], 11, 4)
γ₀₁ = γ[:, 1]
γ₀₂ = γ[:, 2]
γ₀₃ = γ[:, 3]
γ₀₄ = γ[:, 4]

# mount seasonality dummies matrix
D = zeros(n+w, 12)
for t = 1:n+w
    s_t = t % 12 + 1
    D[t, s_t] = 1
end

# apply transformations
ϕ = sigmoid.(ϕ)
m₀ = sigmoid.(m₀)
η = sigmoid.(η)
γ₀₁ = [γ₀₁; -sum(γ₀₁)]
γ₀₂ = [γ₀₂; -sum(γ₀₂)]
γ₀₃ = [γ₀₃; -sum(γ₀₃)]
γ₀₄ = [γ₀₄; -sum(γ₀₄)]
chol_Q = [
    1.0 Q[1] Q[2] Q[3];
    0.0 1.0 Q[4] Q[5];
    0.0 0.0 1.0 Q[6];
    0.0 0.0 0.0 1.0
]
inv_chol_Q = inv(chol_Q)
inv_Q = inv_chol_Q' * inv_chol_Q
Q = chol_Q * chol_Q'
d_m05 = Diagonal(diag(Q) .^ (-0.5))
R = d_m05 * Q * d_m05
v = exp(v) + 2

# initialize variables
m = Matrix{Float64}(undef, n+w, p)
γ = zeros(n+w, 12, p)
μ = Matrix{Float64}(undef, n+w, p)
S = Matrix{Float64}(undef, n+w, p)
ψ = Matrix{Float64}(undef, n+w, p)
V = zeros(n+w, p, p)
Σ = zeros(n+w, p, p)
S2 = Matrix{Float64}(undef, n+w, p)

# loop over time
for t = 1:n

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

    # variance component
    if t == 1
        ψ[t, :] = (1 .- η) .* ρ .+ η .* ψ₀
    else
        ψ[t, :] = (1 .- η) .* ρ .+ η .* ψ[t-1, :] + κ_var .* S2[t-1, :]
    end
    V_t = diagm(exp.(ψ[t, :]))
    Σ[t, :, :] = V_t * R * V_t
    det_Σ = (det(V_t)^2) * prod(diag(Q) .^ (-1))
    inv_Σ = inv(Σ[t, :, :])

    # Sₜ (μ)
    S[t, :] = (
        (v + p) *
        inv(v - 2) *
        (inv_Σ * (Y[t, :] - μ[t, :])) *
        inv(
            1 +
            inv(v - 2) *
            (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :])
        )
    )
    # Sₜ (Σ)
    for i = 1:p
        dV_dϕ = zeros(p, p)
        dV_dϕ[i, i] = 1
        dΣ_dϕ = dV_dϕ * R * V_t + V_t * R * dV_dϕ
        S2[t, i] = (
            (-0.5) *
            sum(diag(inv_Σ * dΣ_dϕ))
        ) + (
            (v + p) *
            inv(2 * (v - 2)) *
            ((Y[t, :] - μ[t, :])'inv_Σ * dΣ_dϕ * inv_Σ * (Y[t, :] - μ[t, :])) *
            inv(
                1 +
                inv(v - 2) *
                (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :])
            )
        )
    end

end

J = 3000
future_Y = []
for j in ProgressBar(1:J)
    for t = n+1:n+w

        m[t, :] = (1 .- ϕ) .* ω + ϕ .* m[t-1, :] + κ_trend .* S[t-1, :]
        aₜ = (-1 / 11) * (ones(12) - 12 * D[t, :])
        γ[t, :, 1] = γ[t-1, :, 1] + aₜ * κ_sazon[1] * S[t-1, 1]
        γ[t, :, 2] = γ[t-1, :, 2] + aₜ * κ_sazon[2] * S[t-1, 2]
        γ[t, :, 3] = γ[t-1, :, 3] + aₜ * κ_sazon[3] * S[t-1, 3]
        γ[t, :, 4] = γ[t-1, :, 4] + aₜ * κ_sazon[4] * S[t-1, 4]
        μ[t, :] = m[t, :] + γ[t, :, :]'D[t, :]
        ψ[t, :] = (1 .- η) .* ρ .+ η .* ψ[t-1, :] + κ_var .* S2[t-1, :]
        V_t = diagm(exp.(ψ[t, :]))
        Σ[t, :, :] = V_t * R * V_t
        det_Σ = (det(V_t)^2) * prod(diag(Q) .^ (-1))
        inv_Σ = inv(Σ[t, :, :])

        dist = MvTDist(exp(v) + 2, μ[t, :], ((v - 2) / v) * round.(Σ[t, :, :], digits=6))
        Y[t, :] = rand(dist)


        # Sₜ (μ)
        S[t, :] = (
            (v + p) *
            inv(v - 2) *
            (inv_Σ * (Y[t, :] - μ[t, :])) *
            inv(
                1 +
                inv(v - 2) *
                (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :])
            )
        )
        # Sₜ (Σ)
        for i = 1:p
            dV_dϕ = zeros(p, p)
            dV_dϕ[i, i] = 1
            dΣ_dϕ = dV_dϕ * R * V_t + V_t * R * dV_dϕ
            S2[t, i] = (
                (-0.5) *
                sum(diag(inv_Σ * dΣ_dϕ))
            ) + (
                (v + p) *
                inv(2 * (v - 2)) *
                ((Y[t, :] - μ[t, :])'inv_Σ * dΣ_dϕ * inv_Σ * (Y[t, :] - μ[t, :])) *
                inv(
                    1 +
                    inv(v - 2) *
                    (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :])
                )
            )
        end
    end
    append!(future_Y, [copy(Y)])
end

using Dates
future_rng = [df[end, 1] + Month(i) for i=1:w]
for i = 1:p
    fig = plot(df[n-120:end, 1], orig_Y[n-120:end, i], color="black", label="Spread Series")
    l = []
    u = []
    for t = n+1:n+w
        append!(l, quantile([future_Y[j][t, i] for j = 1:J], 0.05))
        append!(u, quantile([future_Y[j][t, i] for j = 1:J], 0.95))
    end
    plot!(future_rng, (l .+ u) ./ 2, ribbon=(l .- u) ./ 2, fillalpha=0.35, c=1, label="95% Confidence band")
    savefig(fig, "projeto//ScoreDrivenSpread//data//results//model2//long_forecast_$i.png")
end