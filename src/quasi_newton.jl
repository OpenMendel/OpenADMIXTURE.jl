function update_UV!(U, V, x, x_next, x_next2, iter, Q)
    idx = iter % Q + 1
    println(size(U))
    println(size(x_next2))
    println(size(x_next))
    V[:, idx] .= x_next2 .- x_next
    U[:, idx] .= x_next .- x
end

function update_QN!(x_next, x_mapped, x, U, V)
    x_next .= x_mapped .- V * (inv(U' * (U .- V)) * (U' * (x .- x_mapped)))
end