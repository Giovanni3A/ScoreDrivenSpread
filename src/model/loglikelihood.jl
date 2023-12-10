# declare series size
n = 307
p = 4

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

"""
-Log-likelihood calculation

Parameters
----------
n : Int
    Number of observations
p : Int
    Number of series
θ : Vector{Float}
    All time-fixed parameters

reconstruct : boolean, default False
    If True, returns μ, m and γ series instead of log-likelihood

Returns
-------
L : Float
    (-1) * Log-likelihood value
"""
function loglikelihood(θ, reconstruct=false)
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
    ) = θ
    # aggregate parameters
    m₀ = [m₀₁, m₀₂, m₀₃, m₀₄]
    ω = [ω₁, ω₂, ω₃, ω₄]
    ϕ = [ϕ₁, ϕ₂, ϕ₃, ϕ₄]
    κ_trend = [κ₁, κ₂, κ₃, κ₄]
    κ_sazon = [κ₅, κ₆, κ₇, κ₈]
    γ₀₁ = [γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁]
    γ₀₂ = [γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁]
    γ₀₃ = [γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁]
    γ₀₄ = [γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁]

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
        Σ₁ Σ₂ Σ₃ Σ₄;
        0.0 Σ₅ Σ₆ Σ₇;
        0.0 0.0 Σ₈ Σ₉;
        0.0 0.0 0.0 Σ₁₀
    ]
    inv_Σ = chol_inv_Σ * chol_inv_Σ'
    v = exp(v) + 2

    # initialize variables
    L = 0
    m = Matrix{Float64}(undef, n, p)
    γ = zeros(n, 12, p)
    μ = Matrix{Float64}(undef, n, p)
    S = Matrix{Float64}(undef, n, p)

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

        # log likelihood
        pre_logL = log(gamma((v + p) / 2)) - log(gamma(v / 2)) - 0.5 * p * log(v * π) + log(prod(diag(chol_inv_Σ)))
        L += pre_logL - (v + p) * 0.5 * log(1 + inv(v) * (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :]))
    end
    if reconstruct
        return μ, m, γ
    end
    return -L
end