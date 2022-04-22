using .CUDA
function loglikelihood(g::CuArray{UInt8, 2}, q::CuArray{T, 2}, f::CuArray{T, 2}, I, J) where T
    out = CUDA.zeros(Float64)
    kernel = @cuda launch=false loglikelihood_kernel(out, g, q, f, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    CUDA.@sync kernel(out, g, q, f, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))
    out[]
end

function loglikelihood(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.f, 2)
        return loglikelihood(d.g, d.q, d.f, I, J)
    else
        @assert false "Not implemented"
    end
end

function em_q!(q_next, g::CuArray{UInt8, 2}, 
    q::CuArray{T, 2}, f::CuArray{T, 2}, I, J) where T
    fill!(q_next, zero(eltype(q_next)))
    kernel = @cuda launch=false em_q_kernel!(q_next, g, q, f, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(q_next, g, q, f, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(cld(I, threads_1d), cld(I, threads_1d)))
    q_next ./= 2J
    q_next
end

function em_q!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.f, 2)
        em_q!(d.q_next, g_cu, d.q, d.f, I, J)
    else
        @assert false "Not implemented"
    end   
end

function em_f!(f_next, f_tmp, g::CuArray{UInt8, 2}, 
    q::CuArray{T, 2}, f::CuArray{T, 2}, I, J) where T
    fill!(f_next, zero(eltype(f_next)))
    fill!(f_tmp, zero(eltype(f_tmp)))
    kernel = @cuda launch=false em_f_kernel!(f_next, f_tmp, g, q, f, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(f_next, f_tmp, g, q, f, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(2threads_1d, 2threads_1d))
    f_next .= f_tmp ./ (f_tmp .+ f_next)
    f_next
end

function em_f!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.f, 2)
        em_f!(d.f_next, d.f_tmp, g_cu, d.q, d.f, I, J)
    else
        @assert false "Not implemented"
    end   
end

function update_q_cuda!(XtX, Xtz, g, q, f, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_q_kernel!(XtX, Xtz, g, q, f, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, f, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_q_cuda!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g, 2), size(d.q, 1)
    if J == size(d.f, 2)
        update_q_cuda!(d.XtX_q, d.Xtz_q, g_cu, d.q, d.f, I, J)
    else
        @assert false "Not implemented"
    end   
end

function update_f_cuda!(XtX, Xtz, g, q, f, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_f_kernel!(XtX, Xtz, g, q, f, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, f, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_f_cuda!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.f, 2)
        update_f_cuda!(d.XtX_f, d.Xtz_f, g_cu, d.q, d.f, I, J)
    else
        @assert false "Not implemented"
    end   
end
