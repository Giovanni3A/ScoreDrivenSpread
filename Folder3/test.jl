"""
Add varianca-covariance and df
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions
using StatsBase
using Optim
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

# compute covariance matrix
C = cholesky(cov(Y))
Y_Σ = Matrix(C.U)
Y_Σ = [
    Y_Σ[1, 1], Y_Σ[1, 2],
    Y_Σ[2, 2]
]

# mount seasonality dummies matrix
D = zeros(n, 12)
for t = 1:n
    s_t = t % 12 + 1
    D[t, s_t] = 1
end

# loss function
function loglikelihood(θ, reconstruct=false)
    (
        m₀₁, m₀₂,
        ω₁, ω₂,
        ϕ₁, ϕ₂,
        κ₁, κ₂, κ₃, κ₄,
        γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁, γ₁₁₂,
        γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁, γ₂₁₂,
        Σ₁, Σ₂, Σ₃,
        v
    ) = θ
    m₀ = [m₀₁, m₀₂]
    ω = [ω₁, ω₂]
    ϕ = [ϕ₁, ϕ₂]
    ϕ = sigmoid.(ϕ)
    κ_trend = [κ₁, κ₂]
    κ_sazon = [κ₃, κ₄]
    γ₀₁ = [γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁, γ₁₁₂]
    γ₀₁ = γ₀₁ .- mean(γ₀₁)
    γ₀₂ = [γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁, γ₂₁₂]
    γ₀₂ = γ₀₂ .- mean(γ₀₂)
    Σ = [
        Σ₁ Σ₂;
        0.0 Σ₃
    ]
    Σ = Σ' * Σ
    Σ = (Σ + Σ') / 2
    v = exp(v)

    total = 0
    μ_vals = Matrix{Any}(undef, n, p)
    m_vals = Matrix{Any}(undef, n, p)
    γ_vals = Matrix{Any}(undef, n, p)
    for t = 1:n

        # trend component
        if t == 1
            global m = (1 .- ϕ) .* ω .+ ϕ .* m₀
        else
            global m = (1 .- ϕ) .* ω + ϕ .* m + κ_trend .* S
        end

        # seasonality component
        aₜ = (-1 / 11) * (ones(12) - 12 * D[1, :])
        if t == 1
            global γ₁ = γ₀₁
            global γ₂ = γ₀₂
        else
            global γ₁ = γ₁ + aₜ * κ_sazon[1] * S[1]
            global γ₂ = γ₂ + aₜ * κ_sazon[2] * S[2]
        end

        # μ
        μ = m + [γ₁ γ₂]'D[t, :]

        # score
        global S = (
            -(v + p) *
            inv(v) *
            (inv(Σ) * (Y[t, :] - μ)) *
            inv(
                1 +
                inv(v) *
                (Y[t, :] - μ)' * inv(Σ) * (Y[t, :] - μ)
            )
        )

        # log likelihood
        dist = MvTDist(
            v,
            μ,
            Σ
        )
        total += log(pdf(dist, Y[t, :]))

        if reconstruct
            μ_vals[t, :] = μ
            m_vals[t, :] = m
            γ_vals[t, :] = [γ₁ γ₂]'D[t, :]
        end
    end
    if reconstruct
        return μ_vals, m_vals, γ_vals
    end
    return -total
end

# initial values
initial = [
    Y[1, 1], Y[1, 2],
    mean(Y[:, 1]), mean(Y[:, 2]),
    0.5, 0.5,
    0.5 * ones(p)...,
    0.5 * ones(p)...,
    (D' * Y[:, 1] ./ n)...,
    (D' * Y[:, 2] ./ n)...,
    Y_Σ...,
    3
]
μ_initial, m_initial, γ_initial = loglikelihood(initial, true)
plot(Y[:, 1], label="y₁", color="black")
plot!(μ_initial[:, 1], label="μ₁")
plot!(m_initial[:, 1], label="m₁")
plot!(γ_initial[:, 1], label="γ₁")


# call optimizer
res = optimize(
    loglikelihood,
    initial,
    NelderMead(),
    Optim.Options(
        g_tol=1e-4,
        iterations=999999,
        show_trace=true,
        show_every=1000,
        time_limit=60 * 10
    )
)
solution = Optim.minimizer(res)
(
    m₀₁, m₀₂,
    ω₁, ω₂,
    ϕ₁, ϕ₂,
    κ₁, κ₂, κ₃, κ₄,
    γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁,
    γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁,
    Σ₁, Σ₂, Σ₃,
    v
) = solution

# reconstruct μ
μ_hat, m_hat, γ_hat = loglikelihood(solution, true)

# prediction analysis
plot(Y[:, 1], label="y₁", color="black")
plot!(μ_hat[:, 1], label="μ₁")
plot!(m_hat[:, 1], label="m₁")
plot!(γ_hat[:, 1], label="γ₁")

plot(Y[:, 2], label="y₂", color="black")

plot!(μ_hat[:, 2], label="μ₂")
plot!(m_hat[:, 2], label="m₂")
plot!(γ_hat[:, 2], label="γ₂")

# parameter analysis
println("κ values:")
κ₁, κ₂, κ₃, κ₄
println("ϕ values:")
sigmoid(ϕ₁), sigmoid(ϕ₂)
println("ω values:")
ω₁, ω₂
println("m₀ values:")
m₀₁, m₀₂
println("Σ values:")
Σ = [
    Σ₁ Σ₂;
    0.0 Σ₃
]
Σ = Σ' * Σ
Σ = (Σ + Σ') / 2

# residual analysis
residuals = Y - μ_hat
plot(residuals[:, 1], label="y₁", color="black")
plot!(residuals[:, 2], label="y₂")
bar(autocor(residuals[:, 1], 1:15), label="y₁", color="black")
bar!(autocor(residuals[:, 2], 1:15), label="y₂")

# save results

# create dataframe with parameters
params_df = DataFrame(
    name=[
        "m01", "m02", "m03", "m04",
        "omega1", "omega2", "omega3", "omega4",
        "phi1", "phi2", "phi3", "phi4",
        "kappa1", "kappa2", "kappa3", "kappa4", "kappa5", "kappa6", "kappa7", "kappa8",
        "gamma_1_1", "gamma_1_2", "gamma_1_3", "gamma_1_4", "gamma_1_5", "gamma_1_6", "gamma_1_7", "gamma_1_8", "gamma_1_9", "gamma_1_10", "gamma_1_11",
        "gamma_2_1", "gamma_2_2", "gamma_2_3", "gamma_2_4", "gamma_2_5", "gamma_2_6", "gamma_2_7", "gamma_2_8", "gamma_2_9", "gamma_2_10", "gamma_2_11",
        "gamma_3_1", "gamma_3_2", "gamma_3_3", "gamma_3_4", "gamma_3_5", "gamma_3_6", "gamma_3_7", "gamma_3_8", "gamma_3_9", "gamma_3_10", "gamma_3_11",
        "gamma_4_1", "gamma_4_2", "gamma_4_3", "gamma_4_4", "gamma_4_5", "gamma_4_6", "gamma_4_7", "gamma_4_8", "gamma_4_9", "gamma_4_10", "gamma_4_11",
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
CSV.write("projeto//ScoreDrivenSpread//dados//params.csv", params_df; delim=";", decimal=',')
CSV.write("projeto//ScoreDrivenSpread//dados//μ_hat.csv", μ_hat_df; delim=";", decimal=',')
