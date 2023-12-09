"""
Apply Σ and v
"""

using Distributions, Random
using LinearAlgebra, SpecialFunctions, StatsBase
using JuMP, Ipopt
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
n, p = size(Y)

plot(X[:, 1], Y[:, 1], label="y₁", title="Spread Series")
plot!(X[:, 1], Y[:, 2], label="y₂")
plot!(X[:, 1], Y[:, 3], label="y₃")
plot!(X[:, 1], Y[:, 4], label="y₄")


# mount seasonality dummies matrix
D = zeros(n, 12)
for t = 1:n
    s_t = t % 12 + 1
    D[t, s_t] = 1
end

# utility functions
function invtanh(x)
    return 0.5 * log((1 + x) / (1 - x))
end
function sigmoid(x)
    return 1 / (1 + exp(-x))
end
function logit(x)
    return log(x / (1 - x))
end

# create the optimization model
model = Model(Ipopt.Optimizer)

# declare θ: decision variables
@variables(model, begin
    m₀[1:p]
    ω[1:p]
    logit_ϕ[1:p]
    k[1:p, 1:2]
    flat_chol_invΣ[1:10]
    log_v
end);

# variable transformations
ϕ = sigmoid.(logit_ϕ);
chol_inv_Σ = [
    flat_chol_invΣ[1] flat_chol_invΣ[2] flat_chol_invΣ[3] flat_chol_invΣ[4];
    0 flat_chol_invΣ[5] flat_chol_invΣ[6] flat_chol_invΣ[7];
    0 0 flat_chol_invΣ[8] flat_chol_invΣ[9];
    0 0 0 flat_chol_invΣ[10]
];
inv_Σ = chol_inv_Σ * chol_inv_Σ';
v = exp(log_v) + 2;

m = Matrix{Any}(undef, n + 1, p);
γ = Array{Any}(undef, n + 1, 12, p);
μ = Matrix{Any}(undef, n + 1, p);
S = Matrix{Any}(undef, n + 1, p);
L = 0;

# set initial values
m[1, :] = m₀;
γ[1, :, :] = zeros(12, p);

for t = 1:n

    println(t)

    # μ
    for i = 1:p
        μ[t, i] = m[t, i] + D[t, :]'γ[t, :, i]
    end

    # score
    multi = (inv_Σ * (Y[t, :] .- μ[t, :]))
    for i = 1:p
        S[t, i] = (
            (v + p) * inv(v) * multi[i] *
            inv(1 + inv(v) * sum((Y[t, :] - μ[t, :]) .* multi))
        )
    end

    # trend
    m[t+1, :] = (1 .- ϕ) .* ω + ϕ .* m[t, :] + k[:, 1] .* S[t, :]
    if t == 1
        m[t, :] = (1 .- ϕ) .* ω + ϕ .* m₀
    else
        m[t, :] = (1 .- ϕ) .* ω + ϕ .* m[t-1, :] + k[:, 1] .* S[t-1, :]
    end

    # sazon
    aₜ = (-1 / 11) * (ones(12) - 12 * D[t, :])
    for i = 1:p
        for j = 1:12
            γ[t+1, j, i] = γ[t, j, i] + S[t, i] * k[p, 2] * aₜ[j]
        end
    end

    # log likelihood
    pre_logL = log(gamma((v + p) / 2)) - log(gamma(v / 2)) - 0.5 * p * log(v * π) + log(prod(diag(chol_inv_Σ)))
    L+= pre_logL - (v + p) * 0.5 * log(1 + inv(v) * sum((Y[t, :] - μ[t, :]) .* multi))

end

# set objective value and optimize
@NLobjective(model, Max, L);
