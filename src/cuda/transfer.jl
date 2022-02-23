function copyto_sync!(As_d1::Vector{<:AbstractArray{T}}, As_d2::Vector{<:AbstractArray{T}}) where T
    CUDA.@sync begin
        for (A_d1, A_d2) in zip(As_d1, As_d2)
            copyto!(A_d1, A_d2)
        end
    end
end
