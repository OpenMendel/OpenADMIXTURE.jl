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

    p           ::SAT{T} # K x J
    p_next      ::SAT{T}
    p_next2     ::SAT{T}
    p_tmp       ::SAT{T}

    XtX_q       ::Array{T, 3} # K x K x I
    Xtz_q       ::Matrix{T}   # K x I
    XtX_p       ::Array{T, 3} # K x K x J
    Xtz_p       ::Matrix{T}   # K x J

    # views
    qv          :: OneDSlice{T}
    q_nextv     :: OneDSlice{T}
    q_tmpv      :: OneDSlice{T}
    pv          :: OneDSlice{T}
    p_nextv     :: OneDSlice{T}
    p_tmpv      :: OneDSlice{T}

    XtX_qv      :: TwoDSlice{T}
    Xtz_qv      :: OneDSlice{T}
    XtX_pv      :: TwoDSlice{T}
    Xtz_pv      :: OneDSlice{T}

    qp_small    :: Array{T, 3}   # 64 x 64
    qp_smallv   :: TwoDSlice{T}

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

"""
    AdmixData{T}(I, J, K, Q; skipmissing=true, rng=Random.GLOBAL_RNG)
Constructor for Admixture information.

# Arguments:
- I: Number of samples
- J: Number of variants
- K: Number of clusters
- Q: Number of steps used for quasi-Newton update
- skipmissing: skip computation of loglikelihood for missing values. Should be kept `true` in most cases
- rng: Random number generation.
"""
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

    p       = view(x      , :, (I+1):(I+J))#rand(T, K, J) 
    p       = unsafe_wrap(Array, pointer(p), size(p))
    p_next  = view(x_next , :, (I+1):(I+J))
    p_next  = unsafe_wrap(Array, pointer(p_next), size(p_next))
    p_next2 = view(x_next2, :, (I+1):(I+J))
    p_next2 = unsafe_wrap(Array, pointer(p_next2), size(p_next2))
    p_tmp   = view(x_tmp, :, (I+1):(I+J))
    p_tmp   = unsafe_wrap(Array, pointer(p_tmp), size(p_tmp))

    XtX_q = convert(Array{T}, rand(rng, K, K, I))
    Xtz_q = convert(Array{T}, rand(rng, K, I))

    XtX_p = convert(Array{T}, rand(rng, K, K, J))
    Xtz_p = convert(Array{T}, rand(rng,  K, J))

    qv          = [view(q, :, i) for i in 1:I]
    q_nextv     = [view(q_next, :, i) for i in 1:I]
    q_tmpv     = [view(q_tmp, :, i) for i in 1:I]
    pv          = [view(p, :, j) for j in 1:J]
    p_nextv     = [view(p_next, :, j) for j in 1:J]
    p_tmpv     = [view(p_tmp, :, j) for j in 1:J]

    XtX_qv      = [view(XtX_q, :, :, i) for i in 1:I]
    Xtz_qv      = [view(Xtz_q, :, i) for i in 1:I]
    XtX_pv      = [view(XtX_p, :, :, j) for j in 1:J]
    Xtz_pv      = [view(Xtz_p, :, j) for j in 1:J]

    # q, 
    # q_arr = unsafe_wrap(Array, pointer(q), size(q))
    # q_next_arr = unsafe_wrap(Array, pointer(q_next), size(q_next))
    # q_tmp_arr = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))

    # f_arr = unsafe_wrap(Array, pointer(q), size(q))
    # f_next_arr = unsafe_wrap(Array, pointer(q_next), size(q_next))
    # f_tmp_arr = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))


    # qf = rand(T, I, J);
    maxL = tile_maxiter(typeof(Xtz_p))
    qp_small = convert(Array{T}, rand(rng, maxL, maxL, NT))
    qp_smallv = [view(qp_small, :, :, t) for t in 1:NT]
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
        q, q_next, q_next2, q_tmp, p, p_next, p_next2, p_tmp, 
        XtX_q, Xtz_q, XtX_p, Xtz_p, 
        qv, q_nextv, q_tmpv, pv, p_nextv, p_tmpv, XtX_qv, Xtz_qv, XtX_pv, Xtz_pv,
        qp_small, qp_smallv, U, V, vt, 
        tmp_k, tmp_k1, tmp_k1_, tmp_k2, tmp_k2_, 
        tableau_k1, tableau_k2, swept,
        tmp_kv, tmp_k1v, tmp_k1_v, tmp_k2v, tmp_k2_v,
        tableau_k1v, tableau_k2v, sweptv,
        idx, idxv,
        NaN, NaN)
end
struct QPThreadLocal{T}
    tmp_k   :: Vector{T}
    tmp_k1  :: Vector{T}
    tmp_k1_ :: Vector{T}
    tmp_k2  :: Vector{T}
    tmp_k2_ :: Vector{T}
    tableau_k1 :: Matrix{T}
    tableau_k2 :: Matrix{T}
    swept :: Vector{Bool}
    idx :: Vector{Int}
end
function QPThreadLocal{T}(K::Int) where T
    tmp_k = Vector{T}(undef, K)
    tmp_k1 = Vector{T}(undef, K+1)
    tmp_k1_ = Vector{T}(undef, K+1)
    tmp_k2 = Vector{T}(undef, K+2)
    tmp_k2_ = similar(tmp_k2)
    tableau_k1 = Array{T, 2}(undef, K+1, K+1)
    tableau_k2 = Array{T, 2}(undef, K+2, K+2)
    swept = convert(Vector{Bool}, trues(K))
    idx = Array{Int}(undef, K)
    QPThreadLocal{T}(tmp_k, tmp_k1, tmp_k1_, tmp_k2, tmp_k2_, 
        tableau_k1, tableau_k2, swept, idx)
end