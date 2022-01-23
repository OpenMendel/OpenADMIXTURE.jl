"""
    qf!(qf, q, f)
Computes q * f, in-place.
"""
function qf!(qf, q, f)
    @tullio qf[i, j] = q[k, i] * f[k, j]
end

"""
    loglikelihood(g, qf)
Compute loglikelihood.
"""
function loglikelihood(g::AbstractArray, qf)
    I = size(qf, 1)
    J = size(qf, 2)
    r = 0.0
#     @turbo for j in 1:J
#         for i in 1:I
#             r += g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
#         end
#     end
    @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    r
end

"""
FRAPPE EM algorithm
Assumes qf to be pre-computed.
"""
function em_f!(f_new, g::AbstractArray, q, f, f_tmp, qf) # summation over i: block J side.
    I = size(q, 2)
    J = size(f, 2)
    K = size(f, 1)
    @tullio f_tmp[k, j] = g[i, j] * q[k, i] * f[k, j] / qf[i, j]
    @tullio f_new[k, j] = (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])
    @tullio f_new[k, j] = f_tmp[k, j] / (f_tmp[k, j] + f_new[k, j])
end
function em_q!(q_new, g::AbstractArray, q, f, qf) # summation over j, block I side.
    I = size(q, 2)
    J = size(f, 2)
    K = size(f, 1)
    @tullio q_new[k, i] = (g[i, j] * q[k, i] * f[k, j] / qf[i, j] + 
        (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])) / 2J
end

const tau_schedule = [collect(0.7^i for i in 1:10); 0.01; 0.001; 0.0001; 
    0.00001; 0.000001; 0.0000001; 0.00000001]

function update_q!(q_next, g::AbstractArray, q, f, qdiff, XtX, Xtz, qf)
    I = size(q, 2)
    J = size(f, 2)
    K = size(f, 1)
    prev_ll = likelihood(g, f, q, qf)
    @time @tullio qf[i, j] = q[k, i] * f[k, j]
    @time @tullio XtX[k, k2, i] = g[i, j] / (qf[i, j]) ^ 2 * f[k, j] * f[k2, j] + 
        (2 - g[i, j]) / (1 - qf[i, j]) ^ 2 * (1 - f[k, j]) * (1 - f[k2, j])
    @time @tullio Xtz[k, i] = g[i, j] * f[k, j] / qf[i, j] + 
        (2 - g[i, j]) * (1 - f[k, j]) / (1 - qf[i, j])

    @time begin
        # for QP formulation
        Xtz .*= -1 
        matrix_a = ones(1, K)
        b = [1.0]
        pmin = zeros(K)
        pmax = ones(K)
        @inbounds  for i in 1:I
            XtX_ = view(XtX, :, :, i)
            Xtz_ = view(Xtz, :, i)
            q_ = view(q, :, i)
            qdiff_ = view(qdiff, :, i)
            
            tableau = create_tableau(XtX_, Xtz_, matrix_a, b, q_)
            _, qd = quadratic_program(tableau, q_, pmin, pmax, K, 1) 
            qdiff_ .= qd
        end
        q_next .= q .+ qdiff

        @inbounds for i in 1:I
            project_q!(@view(q_next[:, i]), idx)
        end
    end

    begin
        @time new_ll = likelihood(g, f, q_next, qf)
        tau = 1.0
        for cnt in 1:length(tau_schedule)
            if prev_ll < new_ll
                return q_next
            end
            println(cnt)
            tau = tau_schedule[cnt]
            q_next .= q .+ rho .* qdiff
            @inbounds for i in 1:I
                project_q!(@view(q_next[:, i]), idx)
            end
            new_ll = likelihood(g, f, q_next, qf)
        end
        println(maximum(abs.(qdiff)))
        println("Update failed. Falling back to EM update.")
        em_q!(g, q_next, f, q, qf)
        @inbounds for i in 1:I
            project_q!(@view(q_next[:, i]), idx)
        end
        return q_next
    end
end

function update_f!(f_next, g::AbstractArray{T}, f, q, fdiff, XtX, Xtz, qf) where T
    I = size(q, 2)
    J = size(f, 2)
    K = size(f, 1)
    prev_ll = likelihood(g, f, q, qf)
    @time @tullio qf[i, j] = q[k, i] * f[k, j]
    @time @tullio XtX[k, k2, j] = g[i, j] / (qf[i, j]) ^ 2 * q[k, i] * q[k2, i] + 
            (2 - g[i, j]) / (1 - qf[i, j]) ^ 2 * q[k, i] * q[k2, i]
    @time @tullio Xtz[k, j] = g[i, j] * q[k, i] / qf[i, j] - 
            (2 - g[i, j]) * q[k, i] / (1 - qf[i, j])

    @time begin       
        Xtz .*= -1 
        matrix_a = ones(0, 0)
        b = ones(0)
        pmin = zeros(K)
        pmax = ones(K)
        
        @inbounds for j in 1:J
            XtX_ = view(XtX, :, :, j)
            Xtz_ = view(Xtz, :, j)
            f_ = view(f, :, j)
            fdiff_ = view(fdiff, :, j)
            
            tableau = create_tableau(XtX_, Xtz_, matrix_a, b, f_)
            _, fd = quadratic_program(tableau, f_, pmin, pmax, K, 0) 
            fdiff_ .= fd            
        end
        f_next .= f .+ fdiff
        project_f!(f_next)
    end
    

    begin
        tau = 1.0
        @time new_ll = likelihood(g, f_next, q, qf)
        for cnt in 1:length(tau_schedule)
            if prev_ll < new_ll
                return f_next
            end
            println(cnt)
            tau = tau_schedule[cnt]
            f_next .= f .+ tau .* fdiff
            project_f!(f_next)
            new_ll = likelihood(g, f_next, q, qf)
        end
        println(maximum(abs.(fdiff)))
        println("Update failed. Falling back to EM update.")
        em_f!(g, f_next, f, q, f_tmp, qf)
        project_f!(f_next)
        return f_next 
    end
end
