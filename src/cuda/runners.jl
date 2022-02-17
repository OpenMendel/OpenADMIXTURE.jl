using .CUDA
function loglikelihood_cuda(g, q, f, I, J)
    out = CUDA.zeros(eltype(q))
    kernel = @cuda launch=false loglikelihood_kernel(out, g, q, f, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    CUDA.@sync kernel(out, g, q, f, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))
    out[]
end

function em_q_cuda!(q_next, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, f::AbstractArray{T, 2}, I, J) where T
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

function em_f_cuda!(f_next, f_tmp, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, f::AbstractArray{T, 2}, I, J) where T
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
