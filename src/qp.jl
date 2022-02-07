"""
Calculates the tableau used in minimizing the quadratic 
0.5 x' Q x + r' x, subject to Ax = b and parameter lower 
and upper bounds.
"""
function create_tableau(matrix_q::AbstractMatrix{T}, r::AbstractVector{T},
  matrix_a::AbstractMatrix{T}, b::AbstractVector{T}, x::AbstractVector{T}) where T

  m = size(matrix_a, 1)
  #
  # Create the tableau in the absence of constraints.
  #
  if m == 0
    tableau = [matrix_q (-r); -r' 0]
  else
  #
  # In the presence of constraints compute a constant mu via
  # the Gerschgorin circle theorem so that Q + mu * A' * A
  # is positive definite.
  #
    (matrix_u, d, matrix_v) = svd(matrix_a, full = true)
    matrix_p = matrix_v * matrix_q * matrix_v'
    mu = 0.0
    for i = 1:m
      mu = max((norm(matrix_p[:, i], 1) - 2matrix_p[i, i]) / d[i]^2, mu)
    end
    mu = T(2mu)
    #
    # Now create the tableau.
    #
    tableau = [matrix_q + mu * matrix_a' * matrix_a matrix_a' (-r);
               matrix_a zeros(T, m, m) b - matrix_a * x;
               -r' (b - matrix_a * x)' 0]
  end
  return tableau
end # function create_tableau

"""
Solves the p-dimensional quadratic programming problem
 min [df * delta + 0.5 * delta' * d^2 f * delta]
 subject to: constraint * delta = 0 and pmin <= par + delta <= pmax.
See: Jennrich JI, Sampson PF (1978) "Some problems faced in making
a variance component algorithm into a general mixed model program."
Proceedings of the Eleventh Annual Symposium on the Interface.
Gallant AR, Gerig TM, editors. Institute of Statistics,
North Carolina State University.
"""
function quadratic_program!(delta::AbstractVector{T}, tableau::AbstractMatrix{T}, par::AbstractVector{T},
  pmin::AbstractVector{T}, pmax::AbstractVector{T}, p::Int, c::Int) where T

  # delta = zeros(T, size(par))
  fill!(delta, zero(T))
  #
  # See function create_tableau for the construction of the tableau.
  # For checking tolerance, set diag to the diagonal elements of tableau.
  # Begin by sweeping on those diagonal elements of tableau corresponding
  # to the parameters. Then sweep on the diagonal elements corresponding
  # to the constraints. If any parameter fails the tolerance test, then
  # return and reset the approximate Hessian.
  #
  small = 1e-5
  tol = 1e-8
  d = diag(tableau)
  for i = 1:p
    if d[i] <= 0.0 || tableau[i, i] < d[i] * tol
      return (0, delta)
    else
      sweep!(tableau, i, false)
    end
  end
  swept = trues(p)
  for i = p + 1:p + c
    if tableau[i, i] >= 0.0
      return (0, delta)
    else
      sweep!(tableau, i, false)
    end
  end
  #
  # Take a step in the direction tableau(i, end) for the parameters i
  # that are currently swept. If a boundary is encountered, determine
  # the maximal fractional step possible.
  #
  cycle_main_loop = false
  for iteration = 1:1000
    a = 1.0
    for i = 1:p
      if swept[i]
        ui = tableau[i, end]
        if ui > 0.0
          ai = pmax[i] - par[i] - delta[i]
        else
          ai = pmin[i] - par[i] - delta[i]
        end
        if abs(ui) > 1e-10
          a = min(a, ai / ui)
        end
      end
    end
    #
    # Take the fractional step for the currently swept parameters, and
    # reset the transformed partial derivatives for these parameters.
    #
    for i = 1:p
      if swept[i]
        ui = tableau[i, end]
        delta[i] = delta[i] + a * ui
        tableau[i, end] = (1.0 - a) * ui
        tableau[end, i] = tableau[i, end]
      end
    end
    #
    # Find a swept parameter that is critical, and inverse sweep it.
    # Go back and try to take another step or fractional step.
    #
    cycle_main_loop = false
    for i = 1:p
      critical = pmin[i] >= par[i] + delta[i] - small
      critical = critical || pmax[i]<= par[i] + delta[i] + small
      if swept[i] && abs(tableau[i, i])>1e-10 && critical
        sweep!(tableau, i, true)
        swept[i] = false
        cycle_main_loop = true
        break
      end
    end
    if cycle_main_loop; continue; end
    #
    # Find an unswept parameter that violates the KKT condition
    # and sweep it. Go back and try to take a step or fractional step.
    # If no such parameter exists, then the problem is solved.
    #
    for i = 1:p
      ui = tableau[i, end]
      violation = ui > 0.0 && pmin[i] >= par[i] + delta[i] - small
      violation = violation || (ui<0.0 && pmax[i]<= par[i] + delta[i] + small)
      if !swept[i] && violation
        sweep!(tableau, i, false)
        swept[i] = true
        cycle_main_loop = true
        break
      end
    end
    if cycle_main_loop; continue; end
    return (iteration, delta)
  end
  return (0, delta)
end # function quadratic_program

"""
Sweeps or inverse sweeps the symmetric tableau A on its kth diagonal entry.
"""
function sweep!(matrix_a::AbstractMatrix{T}, k::Int, inverse::Bool = false) where T

  p = 1.0 / matrix_a[k, k]
  v = matrix_a[:, k]
  matrix_a[:, k] .= 0.0
  matrix_a[k, :] .= 0.0
  if inverse
    v[k] = 1.0
  else
    v[k] = -1.0
  end
  for i = 1:size(matrix_a, 1)
    pv = p * v[i] # scalar
    for j in 1:size(matrix_a, 1)
      matrix_a[j, i] = matrix_a[j, i] - pv * v[j]
    end
    # matrix_a[:, i] .= matrix_a[:, i] .- pv * v
  end
end # function sweep!
