function update_UV!(U, V, x, x_next, x_next2, iter, Q)
    idx = (iter - 1) % Q + 1
    V[:, idx] .= x_next2 .- x_next
    U[:, idx] .= x_next .- x
end

function update_QN!(x_next, x_mapped, x, U, V)
    x_next .= x_mapped .- V * (inv(U' * (U .- V)) * (U' * (x .- x_mapped)))
end
