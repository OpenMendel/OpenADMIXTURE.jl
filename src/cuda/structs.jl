struct CuAdmixData{T}
    # g::CuArray{UInt8, 2}
    q::CuArray{T, 2}
    q_next::CuArray{T, 2}
    p::CuArray{T, 2}
    p_next::CuArray{T, 2}
    p_tmp::CuArray{T, 2}
    XtX_q::CuArray{T, 3}
    Xtz_q::CuArray{T, 2}
    XtX_p::CuArray{T, 3}
    Xtz_p::CuArray{T, 2}
end
function CuAdmixData(d::AdmixData{T}, g::SnpLinAlg{T}, width=d.J) where T
    I, J, K = d.I, d.J, d.K
    @assert d.skipmissing "skipmissing must be true for CuAdmixData."
    Ibytes = (I + 3) ÷ 4
    # g_cu = CuArray{UInt8, 2}(undef, Ibytes, width)
    # if size(g, 2) == J
    #     copyto!(g_cu, g.s.data)
    # end
    q = CuArray{T, 2}(undef, K, I)
    q_next = similar(q)
    p = CuArray{T, 2}(undef, K, J)
    p_next = similar(p)
    p_tmp  = similar(p)
    XtX_q = CuArray{T, 3}(undef, K, K, I)
    Xtz_q = CuArray{T, 2}(undef, K, I)
    XtX_p = CuArray{T, 3}(undef, K, K, J)
    Xtz_p = CuArray{T, 2}(undef, K, J)
    CuAdmixData{T}(q, q_next, p, p_next, p_tmp, XtX_q, Xtz_q, XtX_p, Xtz_p)
end

function _cu_admixture_base(d::AdmixData, g_la::SnpLinAlg, I::Int, J::Int)
    d_cu = CuAdmixData(d, g_la)
    Ibytes = (I + 3) ÷ 4
    g_cu = CuArray{UInt8, 2}(undef, Ibytes, J)
    copyto!(g_cu, g_la.s.data)
    d_cu, g_cu
end