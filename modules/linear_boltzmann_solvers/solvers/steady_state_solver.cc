// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/solvers/steady_state_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/ags_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_compute.h"
#include "framework/object_factory.h"
#include "framework/utils/hdf_utils.h"
#include "caliper/cali.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include <memory>

namespace opensn
{

OpenSnRegisterObjectInNamespace(lbs, SteadyStateSolver);

InputParameters
SteadyStateSolver::GetInputParameters()
{
  InputParameters params = Solver::GetInputParameters();

  params.SetGeneralDescription("Implementation of a steady state solver. This solver calls the "
                               "across-groupset (AGS) solver.");
  params.SetDocGroup("LBSExecutors");
  params.ChangeExistingParamToOptional("name", "SteadyStateSolver");
  params.AddRequiredParameter<std::shared_ptr<Problem>>("lbs_problem", "An existing lbs problem");

  return params;
}

std::shared_ptr<SteadyStateSolver>
SteadyStateSolver::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<SteadyStateSolver>("lbs::SteadyStateSolver", params);
}

SteadyStateSolver::SteadyStateSolver(const InputParameters& params)
  : Solver(params), lbs_problem_(params.GetSharedPtrParam<Problem, LBSProblem>("lbs_problem"))
{
}

void
SteadyStateSolver::Initialize()
{
  CALI_CXX_MARK_SCOPE("SteadyStateSolver::Initialize");

  lbs_problem_->Initialize();

  if (not lbs_problem_->GetOptions().read_restart_path.empty())
    ReadRestartData();
}

void
SteadyStateSolver::Execute()
{
auto& options = lbs_problem_->GetOptions();
if (options.phase == "offline")
  {
    CALI_CXX_MARK_SCOPE("SteadyStateSolver::Execute");

    auto& options = lbs_problem_->GetOptions();

    auto& ags_solver = *lbs_problem_->GetAGSSolver();
    ags_solver.Solve();

    if (options.restart_writes_enabled)
      WriteRestartData();

    if (options.use_precursors)
      ComputePrecursors(*lbs_problem_);

    if (options.adjoint)
      lbs_problem_->ReorientAdjointSolution();

    lbs_problem_->UpdateFieldFunctions();

    lbs_problem_->TakeSample(options.param_id);
  }
  if (options.phase == "merge")
  {
    lbs_problem_->MergePhase(options.param_id);
  }
  if (options.phase == "systems")
  {
    lbs_problem_->ReadBasis();
    lbs_problem_->OperatorAction();
    Mat AU_ = lbs_problem_->AssembleAU();
    Vec b_ = lbs_problem_->LoadRHS();
    const std::string& Ar_filename = "rom_system_Ar_" + std::to_string(options.param_id);
    const std::string& rhs_filename = "rom_system_rhs_" + std::to_string(options.param_id);
    lbs_problem_->AssembleROM(AU_, b_, Ar_filename, rhs_filename);
  }
  if (options.phase == "online")
  {
    lbs_problem_->ReadBasis();
    std::vector<double> params = {0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};
    for (const auto& sub_param : params)
    {
      CAROM::Vector* vec = new CAROM::Vector(1, false);
      (*vec)(0) = sub_param;
      lbs_problem_->param_points_.push_back(vec);
    }

    CAROM::Vector* new_point = new CAROM::Vector(1, false);
    (*new_point)(0) = 0.0;
    std::unique_ptr<CAROM::Matrix> Ar_interp;
    std::unique_ptr<CAROM::Vector> rhs_interp;

    lbs_problem_->InterpolateArAndRHS(new_point, Ar_interp, rhs_interp);
    std::cout << "after interpolate\n";
    lbs_problem_->SolveROM(Ar_interp, rhs_interp);
  }
}

bool
SteadyStateSolver::ReadRestartData()
{
  auto& fname = lbs_problem_->GetOptions().read_restart_path;
  auto& phi_old_local = lbs_problem_->GetPhiOldLocal();
  auto& groupsets = lbs_problem_->GetGroupsets();

  auto file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  bool success = (file >= 0);
  if (file >= 0)
  {
    // Read phi
    success &= H5ReadDataset1D<double>(file, "phi_old", phi_old_local);

    // Read psi
    int gs_id = 0;
    for (auto gs : groupsets)
    {
      if (gs.angle_agg)
      {
        std::string name = "delayed_psi_old_gs" + std::to_string(gs_id);
        if (H5Has(file, name))
        {
          std::vector<double> psi;
          success &= H5ReadDataset1D<double>(file, name.c_str(), psi);
          gs.angle_agg->SetOldDelayedAngularDOFsFromSTLVector(psi);
        }
      }
      ++gs_id;
    }

    H5Fclose(file);
  }

  if (success)
    log.Log() << "Successfully read restart data." << std::endl;
  else
    log.Log() << "Failed to read restart data." << std::endl;

  return success;
}

bool
SteadyStateSolver::WriteRestartData()
{
  auto& options = lbs_problem_->GetOptions();
  auto fname = options.write_restart_path;
  auto& phi_old_local = lbs_problem_->GetPhiOldLocal();
  auto& groupsets = lbs_problem_->GetGroupsets();

  auto file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  bool success = (file >= 0);
  if (file >= 0)
  {
    // Write phi
    success &= H5WriteDataset1D<double>(file, "phi_old", phi_old_local);

    // Write psi
    if (options.write_delayed_psi_to_restart)
    {
      int gs_id = 0;
      for (auto gs : lbs_problem_->GetGroupsets())
      {
        if (gs.angle_agg)
        {
          auto psi = gs.angle_agg->GetOldDelayedAngularDOFsAsSTLVector();
          if (not psi.empty())
          {
            std::string name = "delayed_psi_old_gs" + std::to_string(gs_id);
            success &= H5WriteDataset1D<double>(file, name, psi);
          }
        }
        ++gs_id;
      }
    }

    H5Fclose(file);
  }

  if (success)
  {
    lbs_problem_->UpdateRestartWriteTime();
    log.Log() << "Successfully wrote restart data." << std::endl;
  }
  else
    log.Log() << "Failed to write restart data." << std::endl;

  return success;
}

} // namespace opensn
