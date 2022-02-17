struct CuAdmixData{T}
    g::CuArray{Int8, 2}
    q::CuArray{T, 2}
    q_next::CuArray{T, 2}
    f::CuArray{T, 2}
    f_next::CuArray{T, 2}
    f_tmp::CuArray{T, 2}
    XtX_q::CuArray{T, 2}
    Xtz_q::CuArray{T, 2}
    XtX_f::CuArray{T, 2}
    Xtz_f::CuArray{T, 2}
end
function CuAdmixData(g::SnpLinAlg{T}, d::AdmixData{T}, width=d.J) where T
    I, J, K = d.I, d.J, d.K
    Ibytes = (I + 3) ÷ 4
    g_cu = CuArray{Int8, 2}(undef, Ibytes, width)
    if size(g, 2) == J
        copyto!(g_cu, g.s.data)
    end
    q = CuArray{T, 2}(undef, K, I)
    q_next = similar(q)
    f = CuArray{T, 2}(undef, K, J)
    f_next = similar(f)
    f_tmp  = similar(f)
    XtX_q = CuArray{T, 3}(undef, K, K, I)
    Xtz_q = CuArray{T, 2}(undef, K, I)
    XtX_f = CuArray{T, 3}(undef, K, K, J)
    Xtz_f = CuArray{T, 2}(undef, K, J)
    CuAdmixData{T}(g_d, q, q_next, f, f_next, f_tmp, XtX_q, Xtz_q, XtX_f, Xtz_f)
end
