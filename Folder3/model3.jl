"""
Add variance-covariance and df
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions, StatsBase
using Optim
using CSV, DataFrames
using Plots

# read data from monthly_data.csv file
df = CSV.read("projeto//ScoreDrivenSpread//dados//monthly_data.csv", DataFrame)
X = df[188:end-24, :]
y1 = X[:, 7]
y2 = X[:, 8]
y3 = X[:, 9]
y4 = X[:, 10]
Y = [y1 y2 y3 y4]

plot(X[:, 1], Y[:, 1], label="y₁", title="Spread Series")
plot!(X[:, 1], Y[:, 2], label="y₂")
plot!(X[:, 1], Y[:, 3], label="y₃")
plot!(X[:, 1], Y[:, 4], label="y₄")

# get size
n, p = size(Y)

# compute covariance matrix
v = 3
Y_Σ = cov(Y) * (v - 2) / v
Y_Σ = Matrix(cholesky(Y_Σ).U)
Y_Σ = [
    Y_Σ[1, 1], Y_Σ[1, 2], Y_Σ[1, 3], Y_Σ[1, 4],
    Y_Σ[2, 2], Y_Σ[2, 3], Y_Σ[2, 4],
    Y_Σ[3, 3], Y_Σ[3, 4],
    Y_Σ[4, 4]
]

# mount seasonality dummies matrix
D = zeros(n, 12)
for t = 1:n
    s_t = t % 12 + 1
    D[t, s_t] = 1
end

# inverse tanh
function invtanh(x)
    return 0.5 * log((1 + x) / (1 - x))
end

# sigmoid and logit functions
function sigmoid(x)
    return 1 / (1 + exp(-x))
end
function logit(x)
    return log(x / (1 - x))
end

# loss function
function loglikelihood(θ, reconstruct=false)
    (
        m₀₁, m₀₂, m₀₃, m₀₄,
        ω₁, ω₂, ω₃, ω₄,
        ϕ₁, ϕ₂, ϕ₃, ϕ₄,
        κ₁, κ₂, κ₃, κ₄, κ₅, κ₆, κ₇, κ₈,
        γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁, γ₁₁₂,
        γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁, γ₂₁₂,
        γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁, γ₃₁₂,
        γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁, γ₄₁₂,
        Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇, Σ₈, Σ₉, Σ₁₀,
        v
    ) = θ
    m₀ = [m₀₁, m₀₂, m₀₃, m₀₄]
    ω = [ω₁, ω₂, ω₃, ω₄]
    ϕ = [ϕ₁, ϕ₂, ϕ₃, ϕ₄]
    ϕ = sigmoid.(ϕ)
    κ_trend = [κ₁, κ₂, κ₃, κ₄]
    κ_trend = 5 * tanh.(κ_trend)
    κ_sazon = [κ₅, κ₆, κ₇, κ₈]
    κ_sazon = 5 * tanh.(κ_sazon)
    γ₀₁ = [γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁, γ₁₁₂]
    γ₀₁ = γ₀₁ .- mean(γ₀₁)
    γ₀₂ = [γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁, γ₂₁₂]
    γ₀₂ = γ₀₂ .- mean(γ₀₂)
    γ₀₃ = [γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁, γ₃₁₂]
    γ₀₃ = γ₀₃ .- mean(γ₀₃)
    γ₀₄ = [γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁, γ₄₁₂]
    γ₀₄ = γ₀₄ .- mean(γ₀₄)
    Σ = [
        Σ₁ Σ₂ Σ₃ Σ₄;
        0.0 Σ₅ Σ₆ Σ₇;
        0.0 0.0 Σ₈ Σ₉;
        0.0 0.0 0.0 Σ₁₀
    ]
    Σ = Σ' * Σ
    v = exp(v) + 2

    total = 0
    m = Matrix{Float64}(undef, n, p)
    γ = zeros(n, 12, p)
    μ = Matrix{Float64}(undef, n, p)
    S = Matrix{Float64}(undef, n, p)
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

        # μ
        μ[t, :] = m[t, :] + γ[t, :, :]'D[t, :]

        # score
        S[t, :] = (
            (v + p) *
            inv(v) *
            (inv(Σ) * (Y[t, :] - μ[t, :])) *
            inv(
                1 +
                inv(v) *
                (Y[t, :] - μ[t, :])' * inv(Σ) * (Y[t, :] - μ[t, :])
            )
        )

        # log likelihood
        dist = MvTDist(
            v,
            μ[t, :],
            Σ
        )
        total += log(pdf(dist, Y[t, :]))
    end
    if reconstruct
        return μ, m, γ
    end
    return -total
end

# initial values
initial = [
    Y[1, 1], Y[1, 2], Y[1, 3], Y[1, 4],
    mean(Y[:, 1]), mean(Y[:, 2]), mean(Y[:, 3]), mean(Y[:, 4]),
    zeros(p)...,
    1e-3 * ones(p)...,
    0 * ones(p)...,
    1e-1 * D'Y[:, 1]...,
    1e-1 * D'Y[:, 2]...,
    1e-1 * D'Y[:, 3]...,
    1e-1 * D'Y[:, 4]...,
    Y_Σ...,
    0
]

initial_params_df = CSV.read("projeto//ScoreDrivenSpread//Folder3//params.csv", DataFrame; delim=";", decimal=',')
initial = initial_params_df[:, :value]

μ_initial, m_initial, γ_initial = loglikelihood(initial, true);
plot(Y[:, 2], label="y₁", color="black")
plot!(μ_initial[:, 2], label="μ₁")
plot!(m_initial[:, 2], label="m₁")

# call optimizer
res = optimize(
    loglikelihood,
    initial,
    NelderMead(),
    Optim.Options(
        g_tol=1e-5,
        iterations=10,
        show_trace=true,
        show_every=200,
        time_limit=60 * 30
    )
)
solution = Optim.minimizer(res)
(
    m₀₁, m₀₂, m₀₃, m₀₄,
    ω₁, ω₂, ω₃, ω₄,
    ϕ₁, ϕ₂, ϕ₃, ϕ₄,
    κ₁, κ₂, κ₃, κ₄, κ₅, κ₆, κ₇, κ₈,
    γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁,
    γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁,
    γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁,
    γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁,
    Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇, Σ₈, Σ₉, Σ₁₀,
    v
) = solution

# reconstruct μ
μ_hat, m_hat, γ_hat = loglikelihood(solution, true);

# prediction analysis
plot(Y[:, 1], label="y₁", color="black")
plot!(μ_hat[:, 1], label="μ₁")
plot!(m_hat[:, 1], label="m₁")
plot!(μ_hat[:, 1] - m_hat[:, 1], label="γ₁")

plot(Y[:, 2], label="y₂", color="black")
plot!(μ_hat[:, 2], label="μ₂")
plot!(m_hat[:, 2], label="m₂")
plot!(μ_hat[:, 2] - m_hat[:, 2], label="γ₂")

plot(Y[:, 3], label="y₃", color="black")
plot!(μ_hat[:, 3], label="μ₃")
plot!(m_hat[:, 3], label="m₃")
plot!(μ_hat[:, 3] - m_hat[:, 3], label="γ₃")

plot(Y[:, 4], label="y₄", color="black")
plot!(μ_hat[:, 4], label="μ₄")
plot!(m_hat[:, 4], label="m₄")
plot!(μ_hat[:, 4] - m_hat[:, 4], label="γ₄")

# parameter analysis
println("κ values:")
κ₁, κ₂, κ₃, κ₄, κ₅, κ₆, κ₇, κ₈
println("ϕ values:")
sigmoid.([ϕ₁, ϕ₂, ϕ₃, ϕ₄])
println("ω values:")
ω₁, ω₂, ω₃, ω₄
println("m₀ values:")
m₀₁, m₀₂, m₀₃, m₀₄
println("Σ values:")
Σ = [
    Σ₁ Σ₂ Σ₃ Σ₄;
    0.0 Σ₅ Σ₆ Σ₇;
    0.0 0.0 Σ₈ Σ₉;
    0.0 0.0 0.0 Σ₁₀
]
Σ'Σ
println("v value")
exp(v) + 2

# residual analysis
residuals = Y - μ_hat
plot(residuals[:, 1], label="y₁")
plot!(residuals[:, 2], label="y₂")
plot!(residuals[:, 3], label="y₃")
plot!(residuals[:, 4], label="y₄")
bar(autocor(residuals[:, 1], 1:15), label="y₁", color="black")
bar!(autocor(residuals[:, 2], 1:15), label="y₂")
bar!(autocor(residuals[:, 3], 1:15), label="y₃")
bar!(autocor(residuals[:, 4], 1:15), label="y₄")

# save results

# create dataframe with parameters
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
        "sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6", "sigma_7", "sigma_8", "sigma_9", "sigma_10",
        "v"
    ],
    value=solution
)

# create dataframe with μ_hat series
μ_hat_df = DataFrame(
    t=1:n,
    true1=Y[:, 1],
    true2=Y[:, 2],
    true3=Y[:, 3],
    true4=Y[:, 4],
    prev1=μ_hat[:, 1],
    prev2=μ_hat[:, 2],
    prev3=μ_hat[:, 3],
    prev4=μ_hat[:, 4]
)

# save as csv
CSV.write("projeto//ScoreDrivenSpread//Folder3//params.csv", params_df; delim=";", decimal=',')
CSV.write("projeto//ScoreDrivenSpread//Folder3//μ_hat.csv", μ_hat_df; delim=";", decimal=',')
