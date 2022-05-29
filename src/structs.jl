const SAT{T} = Matrix{T}
# const SAT{T} = SubArray{T, 2, Matrix{T}, 
#     Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
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

    # intermediate vectors for LBQN
    x_qq        ::Vector{T}
    x_rr        ::Vector{T}

    q           ::SAT{T} # K x I
    q_next      ::SAT{T}
    q_next2     ::SAT{T}
    q_tmp       ::SAT{T}

    f           ::SAT{T} # K x J
    f_next      ::SAT{T}
    f_next2     ::SAT{T}
    f_tmp       ::SAT{T}

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

    qf_small    :: Array{T, 3}   # 64 x 64
    qf_smallv   :: TwoDSlice{T}

    U           ::Matrix{T}   # K(I + J) x Q
    V           ::Matrix{T}   # K(I + J) x Q

    # for QP
    v_kk        ::Matrix{T}   # K x K, a full svd of ones(1, K)

    tmp_k       ::Matrix{T}
    tmp_k1      ::Matrix{T}
    tmp_k1_      ::Matrix{T}
    tmp_k2      ::Matrix{T}
    tmp_k2_      ::Matrix{T}
    tableau_k1  ::Array{T, 3}   # (K + 1) x (K + 1)
    tableau_k2  ::Array{T, 3}   # (K + 2) x (K + 2)
    swept       ::Matrix{Bool}

    tmp_kv       ::OneDSlice{T}
    tmp_k1v      ::OneDSlice{T}
    tmp_k1_v      ::OneDSlice{T}
    tmp_k2v      ::OneDSlice{T}
    tmp_k2_v      ::OneDSlice{T}
    tableau_k1v  ::TwoDSlice{T}   # (K + 1) x (K + 1)
    tableau_k2v  ::TwoDSlice{T}   # (K + 2) x (K + 2)
    sweptv       ::OneDSlice{Bool}

    idx         ::Matrix{Int}
    idxv        ::OneDSlice{Int}

    # loglikelihoods
    ll_prev     ::Float64
    ll_new      ::Float64
end

function AdmixData{T}(I, J, K, Q; skipmissing=true, rng=Random.GLOBAL_RNG) where T
    NT = nthreads()
    x = convert(Array{T}, rand(rng, K, I + J))
    x_next = similar(x)
    x_next2 = similar(x)
    x_tmp = similar(x)

    x_flat = reshape(x, :)
    x_next_flat = reshape(x_next, :)
    x_next2_flat = reshape(x_next2, :)
    x_tmp_flat = reshape(x_tmp, :)

    x_qq = similar(x_flat)
    x_rr = similar(x_flat)

    q       = view(x      , :, 1:I)#rand(T, K, J) 
    q       = unsafe_wrap(Array, pointer(q), size(q))
    q_next  = view(x_next , :, 1:I)
    q_next  = unsafe_wrap(Array, pointer(q_next), size(q_next))
    q_next2 = view(x_next2, :, 1:I)
    q_next2  = unsafe_wrap(Array, pointer(q_next2), size(q_next2))
    q_tmp   = view(x_tmp, :, 1:I)
    q_tmp  = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))
    q ./= sum(q, dims=1)

    f       = view(x      , :, (I+1):(I+J))#rand(T, K, J) 
    f       = unsafe_wrap(Array, pointer(f), size(f))
    f_next  = view(x_next , :, (I+1):(I+J))
    f_next  = unsafe_wrap(Array, pointer(f_next), size(f_next))
    f_next2 = view(x_next2, :, (I+1):(I+J))
    f_next2 = unsafe_wrap(Array, pointer(f_next2), size(f_next2))
    f_tmp   = view(x_tmp, :, (I+1):(I+J))
    f_tmp   = unsafe_wrap(Array, pointer(f_tmp), size(f_tmp))

    XtX_q = convert(Array{T}, rand(rng, K, K, I))
    Xtz_q = convert(Array{T}, rand(rng, K, I))

    XtX_f = convert(Array{T}, rand(rng, K, K, J))
    Xtz_f = convert(Array{T}, rand(rng,  K, J))

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

    # q, 
    # q_arr = unsafe_wrap(Array, pointer(q), size(q))
    # q_next_arr = unsafe_wrap(Array, pointer(q_next), size(q_next))
    # q_tmp_arr = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))

    # f_arr = unsafe_wrap(Array, pointer(q), size(q))
    # f_next_arr = unsafe_wrap(Array, pointer(q_next), size(q_next))
    # f_tmp_arr = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))


    # qf = rand(T, I, J);
    maxL = tile_maxiter(typeof(Xtz_f))
    qf_small = convert(Array{T}, rand(rng, maxL, maxL, NT))
    qf_smallv = [view(qf_small, :, :, t) for t in 1:NT]
    # qf_thin  = rand(T, I, maxL)
    # f_tmp = similar(f)
    # q_tmp = similar(q);

    V = convert(Array{T}, rand(rng, K * (I+J), Q))
    # V_q = view(reshape(V, K, (I+J), Q), :, 1:I, :)
    # V_f = view(reshape(V, K, (I+J), Q), :, (I+1):(I+J), :)
    U = convert(Array{T}, rand(rng, K * (I+J), Q))

    _, _, vt = svd(ones(T, 1, K), full=true)
    tmp_k = Matrix{T}(undef, K, NT)
    tmp_kv = [view(tmp_k, :, t) for t in 1:NT]
    tmp_k1 = Matrix{T}(undef, K+1, NT)
    tmp_k1v = [view(tmp_k1, :, t) for t in 1:NT]
    tmp_k1_ = similar(tmp_k1)
    tmp_k1_v = [view(tmp_k1_, :, t) for t in 1:NT]
    tmp_k2 = Matrix{T}(undef, K+2, NT)
    tmp_k2v = [view(tmp_k2, :, t) for t in 1:NT]
    tmp_k2_ = similar(tmp_k2)
    tmp_k2_v = [view(tmp_k2_, :, t) for t in 1:NT]
    tableau_k1 = Array{T, 3}(undef, K+1, K+1, NT)
    tableau_k1v = [view(tableau_k1, :, :, t) for t in 1:NT]
    tableau_k2 = Array{T, 3}(undef, K+2, K+2, NT)
    tableau_k2v = [view(tableau_k2, :, :, t) for t in 1:NT]
    swept = convert(Matrix{Bool}, trues(K, NT))
    sweptv = [view(swept, :, t) for t in 1:NT]
    idx = Array{Int}(undef, K, NT)
    idxv = [view(idx, :, t) for t in 1:NT]
    # U_q = view(reshape(U, K, (I+J), Q), :, 1:I, :)
    # U_f = view(reshape(U, K, (I+J), Q), :, (I+1):(I+J), :)

    AdmixData{T}(I, J, K, Q, skipmissing, x, x_next, x_next2, x_tmp, 
        x_flat, x_next_flat, x_next2_flat, x_tmp_flat,
        x_qq, x_rr,
        q, q_next, q_next2, q_tmp, f, f_next, f_next2, f_tmp, 
        XtX_q, Xtz_q, XtX_f, Xtz_f, 
        qv, q_nextv, q_tmpv, fv, f_nextv, f_tmpv, XtX_qv, Xtz_qv, XtX_fv, Xtz_fv,
        qf_small, qf_smallv, U, V, vt, 
        tmp_k, tmp_k1, tmp_k1_, tmp_k2, tmp_k2_, 
        tableau_k1, tableau_k2, swept,
        tmp_kv, tmp_k1v, tmp_k1_v, tmp_k2v, tmp_k2_v,
        tableau_k1v, tableau_k2v, sweptv,
        idx, idxv,
        NaN, NaN)
end
