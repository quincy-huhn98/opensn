#pragma once

#include "framework/math/linear_solver/linear_solver_context.h"

namespace chi_math
{
template <class MatType, class VecType>
int LinearSolverMatrixAction(MatType matrix, VecType vector, VecType action);
} // namespace chi_math