SAT{T} = SubArray{T, 2, Matrix{T}, 
    Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
mutable struct AdmixData{T}
    I           ::Int
    J           ::Int
    K           ::Int
    Q           ::Int

    x           ::Matrix{T} # K x (I + J)
    x_next      ::Matrix{T}
    x_next2     ::Matrix{T}
    x_tmp       ::Matrix{T}

    x_flat      ::Vector{T} # K(I + J)
    x_next_flat ::Vector{T}
    x_next2_flat::Vector{T}
    x_tmp_flat  ::Vector{T}

    q           ::SAT # K x I
    q_next      ::SAT
    q_next2     ::SAT
    q_tmp       ::SAT

    f           ::SAT # K x J
    f_next      ::SAT
    f_next2     ::SAT
    f_tmp       ::SAT

    XtX_q       ::Array{T, 3} # K x K x I
    Xtz_q       ::Matrix{T}   # K x I
    XtX_f       ::Array{T, 3} # K x K x J
    Xtz_f       ::Matrix{T}   # K x J
    qf          ::Matrix{T}   # I x J
    qf_small    ::Matrix{T}   # 64 x 64
    qf_thin     ::Matrix{T}

    U           ::Matrix{T}   # K(I + J) x Q
    V           ::Matrix{T}   # K(I + J) x Q

    ll_prev     ::T
    ll_new      ::T
    idx         ::Array{Int}
    snptmp      ::Array{T}
end

function AdmixData{T}(I, J, K, Q; seed=nothing) where T
    if seed !== nothing
        Random.seed!(seed)
    end
    x = rand(T, K, I + J)
    x_next = similar(x)
    x_next2 = similar(x)
    x_tmp = similar(x)

    x_flat = reshape(x, :)
    x_next_flat = reshape(x_next, :)
    x_next2_flat = reshape(x_next2, :)
    x_tmp_flat = reshape(x_tmp, :)

    q       = view(x      , :, 1:I)#rand(T, K, J) 
    q_next  = view(x_next , :, 1:I)
    q_next2 = view(x_next2, :, 1:I)
    q_tmp   = view(x_tmp, :, 1:I)
    q ./= sum(q, dims=1)

    f       = view(x      , :, (I+1):(I+J))#rand(T, K, J) 
    f_next  = view(x_next , :, (I+1):(I+J))
    f_next2 = view(x_next2, :, (I+1):(I+J))
    f_tmp   = view(x_tmp, :, (I+1):(I+J))

    XtX_q = rand(T, K, K, I)
    Xtz_q = rand(T, K, I)

    XtX_f = rand(T, K, K, J)
    Xtz_f = rand(T, K, J)
    qf = rand(T, I, J);
    maxL = tile_maxiter(typeof(qf))
    qf_small = rand(T, maxL, maxL)
    qf_thin  = rand(T, I, maxL)
    # f_tmp = similar(f)
    # q_tmp = similar(q);

    V = rand(T, K * (I+J), Q)
    # V_q = view(reshape(V, K, (I+J), Q), :, 1:I, :)
    # V_f = view(reshape(V, K, (I+J), Q), :, (I+1):(I+J), :)
    U = rand(T, K * (I+J), Q)
    # U_q = view(reshape(U, K, (I+J), Q), :, 1:I, :)
    # U_f = view(reshape(U, K, (I+J), Q), :, (I+1):(I+J), :)
    snptmp = rand(T, 4)

    AdmixData{T}(I, J, K, Q, x, x_next, x_next2, x_tmp, 
        x_flat, x_next_flat, x_next2_flat, x_tmp_flat,
        q, q_next, q_next2, q_tmp, f, f_next, f_next2, f_tmp, 
        XtX_q, Xtz_q, XtX_f, Xtz_f,
        qf, qf_small, qf_thin, U, V, zero(T), zero(T), Array{Int}(undef, K), snptmp)
end
