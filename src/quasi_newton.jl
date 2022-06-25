"""
    update_UV!(U, V, x, x_next, x_next2, iter, Q)
Update U and V matrices for quasi-Newton acceleration.
"""
function update_UV!(U, V, x, x_next, x_next2, iter, Q)
    idx = (iter - 1) % Q + 1
    V[:, idx] .= x_next2 .- x_next
    U[:, idx] .= x_next .- x
end

"""
    update_UV_LBQN!(U, V, x, x_next, x_next2, iter, Q)
Update U and V matrices for Limited-memory Broyden's quasi-Newton acceleration.
"""
function update_UV_LBQN!(U, V, x, x_next, x_next2, iter, Q)
    @inbounds for i in min(iter, Q):-1:2
        U[:, i] .= @view(U[:, i-1])
        V[:, i] .= @view(V[:, i-1])
    end
    U[:, 1] .= x_next .- x
    V[:, 1] .= x_next2 .- x_next
    V[:, 1] .= @view(V[:, 1]) .- @view(U[:, 1]) # x2 - 2x1 + x0
end

"""
    update_QN!(x_next, x_mapped, x, U, V)
Update one step of quasi-Newton algorithm.
"""
function update_QN!(x_next, x_mapped, x, U, V)
    x_next .= x_mapped .- V * (inv(U' * (U .- V)) * (U' * (x .- x_mapped)))
end

"""
    update_QN_LBQN!(x_next, x, x_q, x_r, U, V)
Update one step of limited-memory broyden's quasi-Newton acceleration.
"""
function update_QN_LBQN!(x_next, x, x_q, x_r, U, V)
    Q = size(U, 2)
    u = @view(U[:, 1])
    v = @view(V[:, 1])

    gamma_t = dot(u, v) / dot(v, v)
    x_q .= u
    alpha = zeros(Q)
    
    @inbounds for i in 1:Q
        rho = 1 / dot(@view(V[:, i]), @view(V[:, i]))
        alpha[i] = rho * dot(@view(V[:, i]), x_q)
        x_q .= x_q .- alpha[i] .* @view(V[:, i])
    end

    x_r .= gamma_t .* x_q
    @inbounds for i in 1:Q
        x_r .= x_r .+ alpha[i] .* @view(U[:, i])
    end

    x_next .= x .- x_r
end
