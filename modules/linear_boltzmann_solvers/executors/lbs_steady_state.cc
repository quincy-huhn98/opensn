// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/executors/lbs_steady_state.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/iterative_methods/ags_solver.h"
#include "framework/object_factory.h"
#include "caliper/cali.h"

namespace opensn
{

OpenSnRegisterObjectInNamespace(lbs, SteadyStateSolver);

InputParameters
SteadyStateSolver::GetInputParameters()
{
  InputParameters params = opensn::Solver::GetInputParameters();

  params.SetGeneralDescription("Implementation of a steady state solver. This solver calls the "
                               "across-groupset (AGS) solver.");
  params.SetDocGroup("LBSExecutors");
  params.ChangeExistingParamToOptional("name", "SteadyStateSolver");
  params.AddRequiredParameter<size_t>("lbs_solver_handle", "Handle to an existing lbs solver");

  return params;
}

SteadyStateSolver::SteadyStateSolver(const InputParameters& params)
  : opensn::Solver(params),
    lbs_solver_(
      GetStackItem<LBSSolver>(object_stack, params.GetParamValue<size_t>("lbs_solver_handle")))
{
}

void
SteadyStateSolver::Initialize()
{
  CALI_CXX_MARK_SCOPE("SteadyStateSolver::Initialize");

  lbs_solver_.Initialize();
}

void
SteadyStateSolver::Execute()
{
  if (lbs_solver_.Options().phase == "offline")
  {
    CALI_CXX_MARK_SCOPE("SteadyStateSolver::Execute");

    auto& ags_solver = *lbs_solver_.GetAGSSolver();
    ags_solver.Solve();

    if (lbs_solver_.Options().use_precursors)
      lbs_solver_.ComputePrecursors();

    if (lbs_solver_.Options().adjoint)
      lbs_solver_.ReorientAdjointSolution();

    lbs_solver_.UpdateFieldFunctions();

    lbs_solver_.TakeSample(lbs_solver_.Options().param_id);
  }
  if (lbs_solver_.Options().phase == "merge")
  {
    lbs_solver_.MergePhase(lbs_solver_.Options().param_id);
  }
  if (lbs_solver_.Options().phase == "online")
  {
    lbs_solver_.ReadBasis();
    lbs_solver_.OperatorAction();
  }
}

} // namespace opensn
