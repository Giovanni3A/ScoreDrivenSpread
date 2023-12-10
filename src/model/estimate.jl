"""
Estimate score driven model parameters
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions, StatsBase
using Optim
using CSV, DataFrames
using Plots
using StatsPlots

# read data from monthly_data.csv file
df = CSV.read("projeto//ScoreDrivenSpread//data//trusted//monthly_data.csv", DataFrame)
X = df[1:end-24, :]  # filter last 24 months
y1 = X[:, 2]
y2 = X[:, 3]
y3 = X[:, 4]
y4 = X[:, 5]
Y = [y1 y2 y3 y4]

plot(X[:, 1], Y[:, 1], label="y₁", title="Spread Series")
plot!(X[:, 1], Y[:, 2], label="y₂")
plot!(X[:, 1], Y[:, 3], label="y₃")
plot!(X[:, 1], Y[:, 4], label="y₄")

# get size
n, p = size(Y)

# include loglikelihood function
include("loglikelihood.jl")

# initial values (from theory)
initial = [
    Y[1, 1], Y[1, 2], Y[1, 3], Y[1, 4],
    mean(Y[:, 1]), mean(Y[:, 2]), mean(Y[:, 3]), mean(Y[:, 4]),
    zeros(p)...,
    1e-3 * ones(p)...,
    0 * ones(p)...,
    rand(10)...,
    0
]
# initial values (from previous estimation)
initial_params_df = CSV.read("projeto//ScoreDrivenSpread//data//results//params.csv", DataFrame; delim=";", decimal=',')
initial = initial_params_df[:, :value]

# initial fitted series
μ_initial, m_initial, γ_initial = loglikelihood(initial, true);
i = 2
plot(Y[:, i], label="y", color="black")
plot!(μ_initial[:, i], label="μ")
plot!(m_initial[:, i], label="m")

# call optimizer
res = optimize(
    loglikelihood,
    initial,
    NelderMead(),
    Optim.Options(
        g_tol=1e-3,
        iterations=10_000,
        show_trace=true,
        show_every=500,
        time_limit=60 * 10
    )
)

# get solution
solution = Optim.minimizer(res)
(
    m₀₁, m₀₂, m₀₃, m₀₄,
    ω₁, ω₂, ω₃, ω₄,
    ϕ₁, ϕ₂, ϕ₃, ϕ₄,
    κ₁, κ₂, κ₃, κ₄, κ₅, κ₆, κ₇, κ₈,
    Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇, Σ₈, Σ₉, Σ₁₀,
    v,
    γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁,
    γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁,
    γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁,
    γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁
) = solution

# reconstruct fitted series
μ_hat, m_hat, γ_hat = loglikelihood(solution, true);

# fit analysis
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
sigmoid.([m₀₁, m₀₂, m₀₃, m₀₄])
println("Σ values:")
Σ = [
    Σ₁ Σ₂ Σ₃ Σ₄;
    0.0 Σ₅ Σ₆ Σ₇;
    0.0 0.0 Σ₈ Σ₉;
    0.0 0.0 0.0 Σ₁₀
];
inv_Σ = Σ'Σ
cov_Y = inv(inv_Σ)
println("v value")
exp(v) + 2

# residual analysis
residuals = Y - μ_hat
plot(residuals[:, 1], label="y₁")
plot!(residuals[:, 2], label="y₂")
plot!(residuals[:, 3], label="y₃")
plot!(residuals[:, 4], label="y₄")
bar(autocor(residuals[:, 1], 1:15), label="y₁", alpha=0.5)
bar!(autocor(residuals[:, 2], 1:15), label="y₂", alpha=0.5)
bar!(autocor(residuals[:, 3], 1:15), label="y₃", alpha=0.5)
bar!(autocor(residuals[:, 4], 1:15), label="y₄", alpha=0.5)

# quantile residual using monte carlo integration
J = 30000
Z = Vector{Float64}(undef, n)
for t = 1:n
    dist = MvTDist(exp(v) + 2, μ_hat[t, :], round.(cov_Y, digits=6))
    samp = rand(dist, J)
    cdf_ = 0
    for j = 1:J
        cdf_ += Int(samp[:, j] <= Y[t, :]) / J
    end
    Z[t] = quantile(Normal(0, 1), cdf_)
end
fig = histogram(Z, title="Histogram of quantile residuals", alpha=0.7)
savefig(fig, "projeto//ScoreDrivenSpread//data//results//quantile_residuals_hist.png")
fig = qqplot(Z, Normal(0, 1), title="QQ-plot of quantile residuals")
savefig(fig, "projeto//ScoreDrivenSpread//data//results//quantile_residuals_qqplot.png")

# create dataframe with parameters (from function)
parameters_names = [
    "m01", "m02", "m03", "m04",
    "omega1", "omega2", "omega3", "omega4",
    "phi1", "phi2", "phi3", "phi4",
    "kappa1", "kappa2", "kappa3", "kappa4", "kappa5", "kappa6", "kappa7", "kappa8",
    "sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6", "sigma_7", "sigma_8", "sigma_9", "sigma_10",
    "v",
    "gamma_1_1", "gamma_1_2", "gamma_1_3", "gamma_1_4", "gamma_1_5", "gamma_1_6", "gamma_1_7", "gamma_1_8", "gamma_1_9", "gamma_1_10", "gamma_1_11",
    "gamma_2_1", "gamma_2_2", "gamma_2_3", "gamma_2_4", "gamma_2_5", "gamma_2_6", "gamma_2_7", "gamma_2_8", "gamma_2_9", "gamma_2_10", "gamma_2_11",
    "gamma_3_1", "gamma_3_2", "gamma_3_3", "gamma_3_4", "gamma_3_5", "gamma_3_6", "gamma_3_7", "gamma_3_8", "gamma_3_9", "gamma_3_10", "gamma_3_11",
    "gamma_4_1", "gamma_4_2", "gamma_4_3", "gamma_4_4", "gamma_4_5", "gamma_4_6", "gamma_4_7", "gamma_4_8", "gamma_4_9", "gamma_4_10", "gamma_4_11"
]
params_df = DataFrame(
    name=parameters_names,
    value=solution
)

# create dataframe with parameters (after transformations)
# re-transform sazonality values
γ₀₁ = [γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁]
γ₀₂ = [γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁]
γ₀₃ = [γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁]
γ₀₄ = [γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁]
γ₀₁ = [γ₀₁; -sum(γ₀₁)]
γ₀₂ = [γ₀₂; -sum(γ₀₂)]
γ₀₃ = [γ₀₃; -sum(γ₀₃)]
γ₀₄ = [γ₀₄; -sum(γ₀₄)]
total_parameters_names = [
    "m01", "m02", "m03", "m04",
    "omega1", "omega2", "omega3", "omega4",
    "phi1", "phi2", "phi3", "phi4",
    "kappa1", "kappa2", "kappa3", "kappa4", "kappa5", "kappa6", "kappa7", "kappa8",
    "sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6", "sigma_7", "sigma_8", "sigma_9", "sigma_10",
    "v",
    "gamma_1_1", "gamma_1_2", "gamma_1_3", "gamma_1_4", "gamma_1_5", "gamma_1_6", "gamma_1_7", "gamma_1_8", "gamma_1_9", "gamma_1_10", "gamma_1_11", "gamma_1_12",
    "gamma_2_1", "gamma_2_2", "gamma_2_3", "gamma_2_4", "gamma_2_5", "gamma_2_6", "gamma_2_7", "gamma_2_8", "gamma_2_9", "gamma_2_10", "gamma_2_11", "gamma_2_12",
    "gamma_3_1", "gamma_3_2", "gamma_3_3", "gamma_3_4", "gamma_3_5", "gamma_3_6", "gamma_3_7", "gamma_3_8", "gamma_3_9", "gamma_3_10", "gamma_3_11", "gamma_3_12",
    "gamma_4_1", "gamma_4_2", "gamma_4_3", "gamma_4_4", "gamma_4_5", "gamma_4_6", "gamma_4_7", "gamma_4_8", "gamma_4_9", "gamma_4_10", "gamma_4_11", "gamma_4_12"
]
transf_params_df = DataFrame(
    name=total_parameters_names,
    value=[
        sigmoid.([m₀₁, m₀₂, m₀₃, m₀₄])...,
        ω₁, ω₂, ω₃, ω₄,
        sigmoid.([ϕ₁, ϕ₂, ϕ₃, ϕ₄])...,
        κ₁, κ₂, κ₃, κ₄, κ₅, κ₆, κ₇, κ₈,
        cov_Y[1, 1], cov_Y[1, 2], cov_Y[1, 3], cov_Y[1, 4], cov_Y[2, 2], cov_Y[2, 3], cov_Y[2, 4], cov_Y[3, 3], cov_Y[3, 4], cov_Y[4, 4],
        exp(v) + 2,
        γ₀₁..., γ₀₂..., γ₀₃..., γ₀₄...
    ]
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
    prev4=μ_hat[:, 4],
    m1=m_hat[:, 1],
    m2=m_hat[:, 2],
    m3=m_hat[:, 3],
    m4=m_hat[:, 4]
)

# save as csv
CSV.write("projeto//ScoreDrivenSpread//data//results//params.csv", params_df; delim=";", decimal=',')
CSV.write("projeto//ScoreDrivenSpread//data//results//total_params.csv", transf_params_df; delim=";", decimal=',')
CSV.write("projeto//ScoreDrivenSpread//data//results//μ_hat.csv", μ_hat_df; delim=";", decimal=',')
