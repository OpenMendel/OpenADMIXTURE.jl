const SAT{T} = SubArray{T, 2, Matrix{T}, 
    Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
const TwoDSlice{T} = Vector{SubArray{T, 2, Array{T, 3}, 
    Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}}
const OneDSlice{T} = Vector{SubArray{T, 1, Matrix{T}, 
    Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}

mutable struct AdmixData{T}
    I           ::Int
    J           ::Int
    K           ::Int
    Q           ::Int
    skipmissing ::Bool

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

    # views
    qv          :: OneDSlice{T}
    q_nextv     :: OneDSlice{T}
    q_tmpv      :: OneDSlice{T}
    fv          :: OneDSlice{T}
    f_nextv     :: OneDSlice{T}
    f_tmpv      :: OneDSlice{T}

    XtX_qv      :: TwoDSlice{T}
    Xtz_qv      :: OneDSlice{T}
    XtX_fv      :: TwoDSlice{T}
    Xtz_fv      :: OneDSlice{T}

    qf_small    ::Matrix{T}   # 64 x 64

    U           ::Matrix{T}   # K(I + J) x Q
    V           ::Matrix{T}   # K(I + J) x Q

    # for QP
    v_kk        ::Matrix{T}   # K x K, a full svd of ones(1, K)
    tmp_k       ::Vector{T}
    tmp_k1      ::Vector{T}
    tmp_k1_      ::Vector{T}
    tmp_k2      ::Vector{T}
    tmp_k2_      ::Vector{T}
    tableau_k1  ::Matrix{T}   # (K + 1) x (K + 1)
    tableau_k2  ::Matrix{T}   # (K + 2) x (K + 2)
    swept       ::BitVector

    ll_prev     ::T
    ll_new      ::T
    idx         ::Array{Int}
    snptmp      ::Array{T}
end

function AdmixData{T}(I, J, K, Q; skipmissing=true, seed=nothing) where T
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

    qv          = [view(q, :, i) for i in 1:I]
    q_nextv     = [view(q_next, :, i) for i in 1:I]
    q_tmpv     = [view(q_tmp, :, i) for i in 1:I]
    fv          = [view(f, :, j) for j in 1:J]
    f_nextv     = [view(f_next, :, j) for j in 1:J]
    f_tmpv     = [view(f_tmp, :, j) for j in 1:J]

    XtX_qv      = [view(XtX_q, :, :, i) for i in 1:I]
    Xtz_qv      = [view(Xtz_q, :, i) for i in 1:I]
    XtX_fv      = [view(XtX_f, :, :, j) for j in 1:J]
    Xtz_fv      = [view(Xtz_f, :, j) for j in 1:J]


    # qf = rand(T, I, J);
    maxL = tile_maxiter(typeof(Xtz_f))
    qf_small = rand(T, maxL, maxL)
    # qf_thin  = rand(T, I, maxL)
    # f_tmp = similar(f)
    # q_tmp = similar(q);

    V = rand(T, K * (I+J), Q)
    # V_q = view(reshape(V, K, (I+J), Q), :, 1:I, :)
    # V_f = view(reshape(V, K, (I+J), Q), :, (I+1):(I+J), :)
    U = rand(T, K * (I+J), Q)

    _, _, vt = svd(ones(T, 1, K), full=true)
    tmp_k = Vector{T}(undef, K)
    tmp_k1 = Vector{T}(undef, K+1)
    tmp_k1_ = similar(tmp_k1)
    tmp_k2 = Vector{T}(undef, K+2)
    tmp_k2_ = similar(tmp_k2)
    tableau_k1 = Matrix{T}(undef, K+1, K+1)
    tableau_k2 = Matrix{T}(undef, K+2, K+2)
    swept = trues(K)
    # U_q = view(reshape(U, K, (I+J), Q), :, 1:I, :)
    # U_f = view(reshape(U, K, (I+J), Q), :, (I+1):(I+J), :)
    snptmp = rand(T, 4)

    AdmixData{T}(I, J, K, Q, skipmissing, x, x_next, x_next2, x_tmp, 
        x_flat, x_next_flat, x_next2_flat, x_tmp_flat,
        q, q_next, q_next2, q_tmp, f, f_next, f_next2, f_tmp, 
        XtX_q, Xtz_q, XtX_f, Xtz_f, 
        qv, q_nextv, q_tmpv, fv, f_nextv, f_tmpv, XtX_qv, Xtz_qv, XtX_fv, Xtz_fv,
        qf_small, U, V, vt, 
        tmp_k, tmp_k1, tmp_k1_, tmp_k2, tmp_k2_, 
        tableau_k1, tableau_k2, swept,
        zero(T), zero(T), Array{Int}(undef, K), snptmp)
end
