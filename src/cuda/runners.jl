using .CUDA
function loglikelihood(g::CuArray{UInt8, 2}, q::CuArray{T, 2}, p::CuArray{T, 2}, I, J) where T
    out = CUDA.zeros(Float64)
    kernel = @cuda launch=false loglikelihood_kernel(out, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    CUDA.@sync kernel(out, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))
    out[]
end

function loglikelihood(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.p, 2)
        return loglikelihood(g_cu, d.q, d.p, I, J)
    else
        @assert false "Not implemented"
    end
end

function em_q!(q_next, g::CuArray{UInt8, 2}, 
    q::CuArray{T, 2}, p::CuArray{T, 2}, I, J) where T
    fill!(q_next, zero(eltype(q_next)))
    kernel = @cuda launch=false em_q_kernel!(q_next, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(q_next, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(cld(I, threads_1d), cld(I, threads_1d)))
    q_next ./= 2J
    q_next
end

function em_q!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.p, 2)
        em_q!(d.q_next, g_cu, d.q, d.p, I, J)
    else
        @assert false "Not implemented"
    end   
end

function em_p!(p_next, p_tmp, g::CuArray{UInt8, 2}, 
    q::CuArray{T, 2}, p::CuArray{T, 2}, I, J) where T
    fill!(p_next, zero(eltype(p_next)))
    fill!(p_tmp, zero(eltype(p_tmp)))
    kernel = @cuda launch=false em_p_kernel!(p_next, p_tmp, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(p_next, p_tmp, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(2threads_1d, 2threads_1d))
    p_next .= p_tmp ./ (p_tmp .+ p_next)
    p_next
end

function em_p!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.p, 2)
        em_p!(d.p_next, d.p_tmp, g_cu, d.q, d.p, I, J)
    else
        @assert false "Not implemented"
    end   
end

function update_q_cuda!(XtX, Xtz, g, q, p, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_q_kernel!(XtX, Xtz, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_q_cuda_gradonly!(XtX, Xtz, g, q, p, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_q_kernel_gradonly!(XtX, Xtz, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_q_cuda!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.p, 2)
        if !d.approxhess
            update_q_cuda!(d.XtX_q, d.Xtz_q, g_cu, d.q, d.p, I, J)
        else
            update_q_cuda_gradonly!(d.XtX_q, d.Xtz_q, g_cu, d.q, d.p, I, J)
        end
    else
        @assert false "Not implemented"
    end   
end

function update_p_cuda!(XtX, Xtz, g, q, p, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_p_kernel!(XtX, Xtz, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_p_cuda_gradonly!(XtX, Xtz, g, q, p, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_p_kernel_gradonly!(XtX, Xtz, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_p_cuda!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = size(d.q, 2), size(g_cu, 2), size(d.q, 1)
    if J == size(d.p, 2)
        if !d.approxhess
            update_p_cuda!(d.XtX_p, d.Xtz_p, g_cu, d.q, d.p, I, J)
        else
            update_p_cuda_gradonly!(d.XtX_p, d.Xtz_p, g_cu, d.q, d.p, I, J)
        end
    else
        @assert false "Not implemented"
    end   
end
