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
        ψ₁, ψ₂, ψ₃, ψ₄,
        ρ₁, ρ₂, ρ₃, ρ₄,
        η₁, η₂, η₃, η₄,
        κ₁, κ₂, κ₃, κ₄, κ₅, κ₆, κ₇, κ₈, κ₉, κ₁₀, κ₁₁, κ₁₂,
        Q₁, Q₂, Q₃, Q₄, Q₅, Q₆,
        v,
        γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁,
        γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁,
        γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁,
        γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁
    ) = θ
    # aggregate mean (μ) parameters
    m₀ = [m₀₁, m₀₂, m₀₃, m₀₄]
    ω = [ω₁, ω₂, ω₃, ω₄]
    ϕ = [ϕ₁, ϕ₂, ϕ₃, ϕ₄]
    κ_trend = [κ₁, κ₂, κ₃, κ₄]
    κ_sazon = [κ₅, κ₆, κ₇, κ₈]
    γ₀₁ = [γ₁₁, γ₁₂, γ₁₃, γ₁₄, γ₁₅, γ₁₆, γ₁₇, γ₁₈, γ₁₉, γ₁₁₀, γ₁₁₁]
    γ₀₂ = [γ₂₁, γ₂₂, γ₂₃, γ₂₄, γ₂₅, γ₂₆, γ₂₇, γ₂₈, γ₂₉, γ₂₁₀, γ₂₁₁]
    γ₀₃ = [γ₃₁, γ₃₂, γ₃₃, γ₃₄, γ₃₅, γ₃₆, γ₃₇, γ₃₈, γ₃₉, γ₃₁₀, γ₃₁₁]
    γ₀₄ = [γ₄₁, γ₄₂, γ₄₃, γ₄₄, γ₄₅, γ₄₆, γ₄₇, γ₄₈, γ₄₉, γ₄₁₀, γ₄₁₁]

    # aggregate variance (Σ) parameters
    ψ₀ = [ψ₁, ψ₂, ψ₃, ψ₄]
    ρ = [ρ₁, ρ₂, ρ₃, ρ₄]
    η = [η₁, η₂, η₃, η₄]
    κ_var = [κ₉, κ₁₀, κ₁₁, κ₁₂]
    chol_Q = [
        1.0 Q₁ Q₂ Q₃;
        0.0 1.0 Q₄ Q₅;
        0.0 0.0 1.0 Q₆;
        0.0 0.0 0.0 1.0
    ]
    inv_chol_Q = inv(chol_Q)
    inv_Q = inv_chol_Q' * inv_chol_Q
    Q = chol_Q * chol_Q'
    d_m05 = Diagonal(diag(Q) .^ (-0.5))
    R = d_m05 * Q * d_m05

    # det(Σ) = det(V)² . det(R)
    # det(R) = prod(diag(Q))^-1

    # apply transformations
    ϕ = sigmoid.(ϕ)
    m₀ = sigmoid.(m₀)
    γ₀₁ = [γ₀₁; -sum(γ₀₁)]
    γ₀₂ = [γ₀₂; -sum(γ₀₂)]
    γ₀₃ = [γ₀₃; -sum(γ₀₃)]
    γ₀₄ = [γ₀₄; -sum(γ₀₄)]
    v = exp(v) + 2
    η = sigmoid.(η)

    # initialize variables
    L = 0
    m = Matrix{Float64}(undef, n, p)
    γ = zeros(n, 12, p)
    μ = Matrix{Float64}(undef, n, p)
    S = Matrix{Float64}(undef, n, p)
    ψ = Matrix{Float64}(undef, n, p)
    V = zeros(n, p, p)
    Σ = zeros(n, p, p)
    S2 = Matrix{Float64}(undef, n, p)

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
            dV_dϕ[i, i] = exp(ψ[t, i])
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

        # log likelihood
        pre_logL = log(gamma((v + p) / 2)) - log(gamma(v / 2)) - 0.5 * p * log((v - 2) * π) - log(det_Σ)
        L += pre_logL - (v + p) * 0.5 * log(1 + inv(v - 2) * (Y[t, :] - μ[t, :])' * inv_Σ * (Y[t, :] - μ[t, :]))
    end
    if reconstruct
        return μ, m, γ, Σ
    end
    return -L
end