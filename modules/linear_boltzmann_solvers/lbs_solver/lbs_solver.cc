// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_solver/lbs_solver.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/boundary/reflecting_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/boundary/vacuum_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/boundary/isotropic_boundary.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/iterative_methods/wgs_context.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/iterative_methods/ags_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/acceleration/diffusion_mip_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/groupset/lbs_groupset.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/point_source/point_source.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/materials/multi_group_xs/multi_group_xs.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/math/time_integrations/time_integration.h"
#include "framework/field_functions/field_function_grid_based.h"
#include "framework/materials/material.h"
#include "framework/logging/log.h"
#include "framework/utils/hdf_utils.h"
#include "framework/object_factory.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cassert>
#include <sys/stat.h>

namespace opensn
{

std::map<std::string, uint64_t> LBSSolver::supported_boundary_names = {
  {"xmin", XMIN}, {"xmax", XMAX}, {"ymin", YMIN}, {"ymax", YMAX}, {"zmin", ZMIN}, {"zmax", ZMAX}};

std::map<uint64_t, std::string> LBSSolver::supported_boundary_ids = {
  {XMIN, "xmin"}, {XMAX, "xmax"}, {YMIN, "ymin"}, {YMAX, "ymax"}, {ZMIN, "zmin"}, {ZMAX, "zmax"}};

OpenSnRegisterSyntaxBlockInNamespace(lbs, OptionsBlock, LBSSolver::OptionsBlock);

OpenSnRegisterSyntaxBlockInNamespace(lbs, BoundaryOptionsBlock, LBSSolver::BoundaryOptionsBlock);

LBSSolver::LBSSolver(const std::string& name) : Solver(name)
{
}

InputParameters
LBSSolver::GetInputParameters()
{
  InputParameters params = Solver::GetInputParameters();

  params.ChangeExistingParamToOptional("name", "LBSDatablock");

  params.AddRequiredParameter<size_t>("num_groups", "The total number of groups within the solver");

  params.AddRequiredParameterArray("groupsets",
                                   "An array of blocks each specifying the input parameters for a "
                                   "<TT>LBSGroupset</TT>.");
  params.LinkParameterToBlock("groupsets", "LBSGroupset");

  params.AddOptionalParameterBlock(
    "options", ParameterBlock(), "Block of options. See <TT>OptionsBlock</TT>.");
  params.LinkParameterToBlock("options", "OptionsBlock");

  return params;
}

LBSSolver::LBSSolver(const InputParameters& params) : Solver(params)
{
  // Make groups
  const size_t num_groups = params.GetParamValue<size_t>("num_groups");
  for (size_t g = 0; g < num_groups; ++g)
    groups_.push_back(LBSGroup(static_cast<int>(g)));

  // Make groupsets
  const auto& groupsets_array = params.GetParam("groupsets");

  const size_t num_gs = groupsets_array.NumParameters();
  for (size_t gs = 0; gs < num_gs; ++gs)
  {
    const auto& groupset_params = groupsets_array.GetParam(gs);

    InputParameters gs_input_params = LBSGroupset::GetInputParameters();
    gs_input_params.SetObjectType("LBSSolver:LBSGroupset");
    gs_input_params.AssignParameters(groupset_params);

    groupsets_.emplace_back(gs_input_params, gs, *this);
  }

  // Options
  if (params.ParametersAtAssignment().Has("options"))
  {
    auto options_params = LBSSolver::OptionsBlock();
    options_params.AssignParameters(params.GetParam("options"));

    this->SetOptions(options_params);
  }
}

LBSOptions&
LBSSolver::Options()
{
  return options_;
}

const LBSOptions&
LBSSolver::Options() const
{
  return options_;
}

size_t
LBSSolver::NumMoments() const
{
  return num_moments_;
}

size_t
LBSSolver::NumGroups() const
{
  return num_groups_;
}

size_t
LBSSolver::NumPrecursors() const
{
  return num_precursors_;
}

size_t
LBSSolver::MaxPrecursorsPerMaterial() const
{
  return max_precursors_per_material_;
}

void
LBSSolver::AddGroup(int id)
{
  if (id < 0)
    groups_.emplace_back(static_cast<int>(groups_.size()));
  else
    groups_.emplace_back(id);
}

const std::vector<LBSGroup>&
LBSSolver::Groups() const
{
  return groups_;
}

void
LBSSolver::AddGroupset()
{
  groupsets_.emplace_back(static_cast<int>(groupsets_.size()));
}

std::vector<LBSGroupset>&
LBSSolver::Groupsets()
{
  return groupsets_;
}

const std::vector<LBSGroupset>&
LBSSolver::Groupsets() const
{
  return groupsets_;
}

void
LBSSolver::AddPointSource(PointSource&& point_source)
{
  point_sources_.push_back(std::move(point_source));
}

void
LBSSolver::ClearPointSources()
{
  point_sources_.clear();
}

const std::vector<PointSource>&
LBSSolver::PointSources() const
{
  return point_sources_;
}

void
LBSSolver::AddVolumetricSource(VolumetricSource&& volumetric_source)
{
  volumetric_sources_.push_back(std::move(volumetric_source));
}

void
LBSSolver::ClearVolumetricSources()
{
  volumetric_sources_.clear();
}

const std::vector<VolumetricSource>&
LBSSolver::VolumetricSources() const
{
  return volumetric_sources_;
}

const std::map<int, std::shared_ptr<MultiGroupXS>>&
LBSSolver::GetMatID2XSMap() const
{
  return matid_to_xs_map_;
}

const std::map<int, std::shared_ptr<IsotropicMultiGroupSource>>&
LBSSolver::GetMatID2IsoSrcMap() const
{
  return matid_to_src_map_;
}

const MeshContinuum&
LBSSolver::Grid() const
{
  return *grid_ptr_;
}

const SpatialDiscretization&
LBSSolver::SpatialDiscretization() const
{
  return *discretization_;
}

const std::vector<UnitCellMatrices>&
LBSSolver::GetUnitCellMatrices() const
{
  return unit_cell_matrices_;
}

const std::map<uint64_t, UnitCellMatrices>&
LBSSolver::GetUnitGhostCellMatrices() const
{
  return unit_ghost_cell_matrices_;
}

const std::vector<CellLBSView>&
LBSSolver::GetCellTransportViews() const
{
  return cell_transport_views_;
}

const UnknownManager&
LBSSolver::UnknownManager() const
{
  return flux_moments_uk_man_;
}

size_t
LBSSolver::LocalNodeCount() const
{
  return local_node_count_;
}

size_t
LBSSolver::GlobalNodeCount() const
{
  return glob_node_count_;
}

std::vector<double>&
LBSSolver::QMomentsLocal()
{
  return q_moments_local_;
}

const std::vector<double>&
LBSSolver::QMomentsLocal() const
{
  return q_moments_local_;
}

std::vector<double>&
LBSSolver::ExtSrcMomentsLocal()
{
  return ext_src_moments_local_;
}

const std::vector<double>&
LBSSolver::ExtSrcMomentsLocal() const
{
  return ext_src_moments_local_;
}

std::vector<double>&
LBSSolver::PhiOldLocal()
{
  return phi_old_local_;
}

const std::vector<double>&
LBSSolver::PhiOldLocal() const
{
  return phi_old_local_;
}

std::vector<double>&
LBSSolver::PhiNewLocal()
{
  return phi_new_local_;
}

const std::vector<double>&
LBSSolver::PhiNewLocal() const
{
  return phi_new_local_;
}

std::vector<double>&
LBSSolver::PrecursorsNewLocal()
{
  return precursor_new_local_;
}

const std::vector<double>&
LBSSolver::PrecursorsNewLocal() const
{
  return precursor_new_local_;
}

std::vector<std::vector<double>>&
LBSSolver::PsiNewLocal()
{
  return psi_new_local_;
}

const std::vector<std::vector<double>>&
LBSSolver::PsiNewLocal() const
{
  return psi_new_local_;
}

std::vector<double>&
LBSSolver::DensitiesLocal()
{
  return densities_local_;
}

const std::vector<double>&
LBSSolver::DensitiesLocal() const
{
  return densities_local_;
}

const std::map<uint64_t, std::shared_ptr<SweepBoundary>>&
LBSSolver::SweepBoundaries() const
{
  return sweep_boundaries_;
}

SetSourceFunction
LBSSolver::GetActiveSetSourceFunction() const
{
  return active_set_source_function_;
}

std::shared_ptr<AGSSolver>
LBSSolver::GetAGSSolver()
{
  return ags_solver_;
}

std::vector<std::shared_ptr<LinearSolver>>&
LBSSolver::GetWGSSolvers()
{
  return wgs_solvers_;
}

size_t&
LBSSolver::LastRestartTime()
{
  return last_restart_write_time_;
}

WGSContext&
LBSSolver::GetWGSContext(int groupset_id)
{
  auto& wgs_solver = wgs_solvers_[groupset_id];
  auto raw_context = wgs_solver->GetContext();
  auto wgs_context_ptr = std::dynamic_pointer_cast<WGSContext>(raw_context);
  OpenSnLogicalErrorIf(not wgs_context_ptr, "Failed to cast WGSContext");
  return *wgs_context_ptr;
}

std::map<uint64_t, BoundaryPreference>&
LBSSolver::BoundaryPreferences()
{
  return boundary_preferences_;
}

std::pair<size_t, size_t>
LBSSolver::GetNumPhiIterativeUnknowns()
{
  const auto& sdm = *discretization_;
  const size_t num_local_phi_dofs = sdm.GetNumLocalDOFs(flux_moments_uk_man_);
  const size_t num_globl_phi_dofs = sdm.GetNumGlobalDOFs(flux_moments_uk_man_);

  return {num_local_phi_dofs, num_globl_phi_dofs};
}

size_t
LBSSolver::MapPhiFieldFunction(size_t g, size_t m) const
{
  OpenSnLogicalErrorIf(phi_field_functions_local_map_.count({g, m}) == 0,
                       std::string("Failure to map phi field function g") + std::to_string(g) +
                         " m" + std::to_string(m));

  return phi_field_functions_local_map_.at({g, m});
}

size_t
LBSSolver::GetHandleToPowerGenFieldFunc() const
{
  OpenSnLogicalErrorIf(not options_.power_field_function_on,
                       "Called when options_.power_field_function_on == false");

  return power_gen_fieldfunc_local_handle_;
}

void
LBSSolver::TakeSample(int id)
{
  bool update_right_SV = false;
  CAROM::Options* options;
  CAROM::BasisGenerator *generator;
  int max_num_snapshots = 100;
  bool isIncremental = false;
  const std::string basisName = "basis";
  const std::string basisFileName = basisName + std::to_string(id);

  options = new CAROM::Options(local_node_count_, max_num_snapshots,
                                     update_right_SV);
  generator = new CAROM::BasisGenerator(*options, isIncremental, basisFileName);


  bool addSample = generator->takeSample(phi_new_local_.data());
  generator->writeSnapshot();
  delete generator;
  delete options;
}

void
LBSSolver::MergePhase(int nsnaps)
{
  bool update_right_SV = false;
  CAROM::Options* options;
  CAROM::BasisGenerator *generator;
  int max_num_snapshots = 100;
  bool isIncremental = false;
  const std::string basisName = "basis";

  options = new CAROM::Options(local_node_count_, max_num_snapshots,
                                     update_right_SV);
  generator = new CAROM::BasisGenerator(*options, isIncremental, basisName);

  for (int paramID=0; paramID<nsnaps; ++paramID)
  {
    std::string snapshot_filename = basisName + std::to_string(
                                        paramID) + "_snapshot";
    generator->loadSamples(snapshot_filename,"snapshot");
  }
  generator->endSamples(); // save the merged basis file
  delete generator;
  delete options;
}

InputParameters
LBSSolver::OptionsBlock()
{
  InputParameters params;

  params.SetGeneralDescription("Set options from a large list of parameters");
  params.SetDocGroup("LBSUtilities");
  params.AddOptionalParameter("spatial_discretization",
                              "pwld",
                              "What spatial discretization to use. Currently only `\"pwld\"` "
                              "is supported");
  params.AddOptionalParameter(
    "scattering_order", 1, "Defines the level of harmonic expansion for the scattering source.");
  params.AddOptionalParameter("max_mpi_message_size",
                              32768,
                              "The maximum MPI message size used during sweep initialization.");
  params.AddOptionalParameter(
    "read_restart_path", "", "Full path for reading restart dumps including file stem.");
  params.AddOptionalParameter(
    "write_restart_path", "", "Full path for writing restart dumps including file stem.");
  params.AddOptionalParameter("write_restart_time_interval",
                              0,
                              "Time interval in seconds at which restart data is to be written.");
  params.AddOptionalParameter(
    "use_precursors", false, "Flag for using delayed neutron precursors.");
  params.AddOptionalParameter("use_source_moments",
                              false,
                              "Flag for ignoring fixed sources and selectively using source "
                              "moments obtained elsewhere.");
  params.AddOptionalParameter(
    "save_angular_flux", false, "Flag indicating whether angular fluxes are to be stored or not.");
  params.AddOptionalParameter(
    "adjoint", false, "Flag for toggling whether the solver is in adjoint mode.");
  params.AddOptionalParameter(
    "verbose_inner_iterations", true, "Flag to control verbosity of inner iterations.");
  params.AddOptionalParameter(
    "verbose_outer_iterations", true, "Flag to control verbosity of across-groupset iterations.");
  params.AddOptionalParameter(
    "max_ags_iterations", 100, "Maximum number of across-groupset iterations.");
  params.AddOptionalParameter("ags_tolerance", 1.0e-6, "Across-groupset iterations tolerance.");
  params.AddOptionalParameter("ags_convergence_check",
                              "l2",
                              "Type of convergence check for AGS iterations. Valid values are "
                              "`\"l2\"` and '\"pointwise\"'");
  params.AddOptionalParameter(
    "verbose_ags_iterations", true, "Flag to control verbosity of across-groupset iterations.");
  params.AddOptionalParameter("power_field_function_on",
                              false,
                              "Flag to control the creation of the power generation field "
                              "function. If set to `true` then a field function will be created "
                              "with the general name <solver_name>_power_generation`.");
  params.AddOptionalParameter("power_default_kappa",
                              3.20435e-11,
                              "Default `kappa` value (Energy released per fission) to use for "
                              "power generation when cross sections do not have `kappa` values. "
                              "Default: 3.20435e-11 Joule (corresponding to 200 MeV per fission).");
  params.AddOptionalParameter("power_normalization",
                              -1.0,
                              "Power normalization factor to use. Supply a negative or zero number "
                              "to turn this off.");
  params.AddOptionalParameter("field_function_prefix_option",
                              "prefix",
                              "Prefix option on field function names. Default: `\"prefix\"`. Can "
                              "be `\"prefix\"` or `\"solver_name\"`. By default this option uses "
                              "the value of the `field_function_prefix` parameter. If this "
                              "parameter is not set, flux field functions will be exported as "
                              "`phi_gXXX_mYYY` where `XXX` is the zero padded 3 digit group number "
                              "and `YYY` is the zero padded 3 digit moment.");
  params.AddOptionalParameter("field_function_prefix",
                              "",
                              "Prefix to use on all field functions. Default: `\"\"`. By default "
                              "this option is empty. Ff specified, flux moments will be exported "
                              "as `prefix_phi_gXXX_mYYY` where `XXX` is the zero padded 3 digit "
                              "group number and `YYY` is the zero padded 3 digit moment. The "
                              "underscore after \"prefix\" is added automatically.");
  params.AddOptionalParameterArray(
    "boundary_conditions", {}, "An array containing tables for each boundary specification.");
  params.LinkParameterToBlock("boundary_conditions", "BoundaryOptionsBlock");
  params.AddOptionalParameter("clear_boundary_conditions",
                              false,
                              "Clears all boundary conditions. If no additional boundary "
                              "conditions are supplied, this results in all boundaries being "
                              "vacuum.");
  params.AddOptionalParameterArray("point_sources", {}, "An array of handles to point sources.");
  params.AddOptionalParameter("clear_point_sources", false, "Clears all point sources.");
  params.AddOptionalParameterArray(
    "volumetric_sources", {}, "An array of handles to volumetric sources.");
  params.AddOptionalParameter("clear_volumetric_sources", false, "Clears all volumetric sources.");
  params.ConstrainParameterRange("spatial_discretization", AllowableRangeList::New({"pwld"}));
  params.ConstrainParameterRange("ags_convergence_check",
                                 AllowableRangeList::New({"l2", "pointwise"}));
  params.ConstrainParameterRange("field_function_prefix_option",
                                 AllowableRangeList::New({"prefix", "solver_name"}));
  params.AddOptionalParameter("param_id", 0, "A parameter id for parametric problems.");
  params.AddOptionalParameter("phase", "offline", "The phase (offline, online, or merge) for ROM purposes.");

  return params;
}

InputParameters
LBSSolver::BoundaryOptionsBlock()
{
  InputParameters params;

  params.SetGeneralDescription("Set options for boundary conditions. See \\ref LBSBCs");
  params.SetDocGroup("LBSUtilities");
  params.AddRequiredParameter<std::string>("name",
                                           "Boundary name that identifies the specific boundary");
  params.AddRequiredParameter<std::string>("type", "Boundary type specification.");
  params.AddOptionalParameterArray<double>("group_strength",
                                           {},
                                           "Required only if \"type\" is \"isotropic\". An array "
                                           "of isotropic strength per group");
  params.AddOptionalParameter("function_name",
                              "",
                              "Text name of the lua function to be called for this boundary "
                              "condition.");
  params.ConstrainParameterRange(
    "name", AllowableRangeList::New({"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}));
  params.ConstrainParameterRange("type",
                                 AllowableRangeList::New({"vacuum", "isotropic", "reflecting"}));

  return params;
}

void
LBSSolver::SetOptions(const InputParameters& params)
{
  const auto& user_params = params.ParametersAtAssignment();

  // Handle order sensitive options
  if (user_params.Has("clear_boundary_conditions"))
  {
    if (user_params.GetParamValue<bool>("clear_boundary_conditions"))
      boundary_preferences_.clear();
  }

  if (user_params.Has("clear_point_sources"))
  {
    if (user_params.GetParamValue<bool>("clear_point_sources"))
      point_sources_.clear();
  }

  if (user_params.Has("clear_volumetric_sources"))
  {
    if (user_params.GetParamValue<bool>("clear_volumetric_sources"))
      volumetric_sources_.clear();
  }

  if (user_params.Has("adjoint"))
  {
    const bool adjoint = user_params.GetParamValue<bool>("adjoint");
    if (adjoint != options_.adjoint)
    {
      options_.adjoint = adjoint;

      // If a discretization exists, the solver has already been initialized.
      // Reinitialize the materials to obtain the appropriate xs and clear the
      // sources to prepare for defining the adjoint problem
      if (discretization_)
      {
        // The materials are reinitialized here to ensure that the proper cross sections
        // are available to the solver. Because an adjoint solve requires volumetric or
        // point sources, the material-based sources are not set within the initialize routine.
        InitializeMaterials();

        // Forward and adjoint sources are fundamentally different, so any existing sources
        // should be cleared and reset through options upon changing modes.
        point_sources_.clear();
        volumetric_sources_.clear();
        boundary_preferences_.clear();

        // Set all solutions to zero.
        phi_old_local_.assign(phi_old_local_.size(), 0.0);
        phi_new_local_.assign(phi_new_local_.size(), 0.0);
        for (auto& psi : psi_new_local_)
          psi.assign(psi.size(), 0.0);
        precursor_new_local_.assign(precursor_new_local_.size(), 0.0);
      }
    }
  }

  // Handle order insensitive options
  for (size_t p = 0; p < user_params.NumParameters(); ++p)
  {
    const auto& spec = user_params.GetParam(p);

    if (spec.Name() == "spatial_discretization")
    {
      auto sdm_name = spec.GetValue<std::string>();
      if (sdm_name == "pwld")
        options_.sd_type = SpatialDiscretizationType::PIECEWISE_LINEAR_DISCONTINUOUS;
    }

    else if (spec.Name() == "scattering_order")
      options_.scattering_order = spec.GetValue<int>();

    else if (spec.Name() == "max_mpi_message_size")
      options_.max_mpi_message_size = spec.GetValue<int>();

    else if (spec.Name() == "read_restart_path")
      options_.read_restart_path = spec.GetValue<std::string>();

    else if (spec.Name() == "write_restart_time_interval")
      options_.write_restart_time_interval = spec.GetValue<int>();

    else if (spec.Name() == "write_restart_path")
      options_.write_restart_path = spec.GetValue<std::string>();

    else if (spec.Name() == "use_precursors")
      options_.use_precursors = spec.GetValue<bool>();

    else if (spec.Name() == "use_source_moments")
      options_.use_src_moments = spec.GetValue<bool>();

    else if (spec.Name() == "save_angular_flux")
      options_.save_angular_flux = spec.GetValue<bool>();

    else if (spec.Name() == "verbose_inner_iterations")
      options_.verbose_inner_iterations = spec.GetValue<bool>();

    else if (spec.Name() == "max_ags_iterations")
      options_.max_ags_iterations = spec.GetValue<int>();

    else if (spec.Name() == "ags_tolerance")
      options_.ags_tolerance = spec.GetValue<double>();

    else if (spec.Name() == "ags_convergence_check")
    {
      auto check = spec.GetValue<std::string>();
      if (check == "pointwise")
        options_.ags_pointwise_convergence = true;
    }

    else if (spec.Name() == "verbose_ags_iterations")
      options_.verbose_ags_iterations = spec.GetValue<bool>();

    else if (spec.Name() == "verbose_outer_iterations")
      options_.verbose_outer_iterations = spec.GetValue<bool>();

    else if (spec.Name() == "power_field_function_on")
      options_.power_field_function_on = spec.GetValue<bool>();

    else if (spec.Name() == "power_default_kappa")
      options_.power_default_kappa = spec.GetValue<double>();

    else if (spec.Name() == "power_normalization")
      options_.power_normalization = spec.GetValue<double>();

    else if (spec.Name() == "field_function_prefix_option")
    {
      options_.field_function_prefix_option = spec.GetValue<std::string>();
    }

    else if (spec.Name() == "field_function_prefix")
      options_.field_function_prefix = spec.GetValue<std::string>();

    else if (spec.Name() == "boundary_conditions")
    {
      spec.RequireBlockTypeIs(ParameterBlockType::ARRAY);
      for (size_t b = 0; b < spec.NumParameters(); ++b)
      {
        auto bndry_params = BoundaryOptionsBlock();
        bndry_params.AssignParameters(spec.GetParam(b));
        SetBoundaryOptions(bndry_params);
      }

      // If a discretization exists, initialize the boundaries.
      if (discretization_)
        InitializeBoundaries();
    }

    else if (spec.Name() == "point_sources")
    {
      spec.RequireBlockTypeIs(ParameterBlockType::ARRAY);
      for (const auto& sub_param : spec)
      {
        point_sources_.push_back(
          GetStackItem<PointSource>(object_stack, sub_param.GetValue<size_t>(), __FUNCTION__));

        // If a discretization exists, the point source can be initialized.
        if (discretization_)
          point_sources_.back().Initialize(*this);
      }
    }

    else if (spec.Name() == "volumetric_sources")
    {
      spec.RequireBlockTypeIs(ParameterBlockType::ARRAY);
      for (const auto& sub_param : spec)
      {
        volumetric_sources_.push_back(
          GetStackItem<VolumetricSource>(object_stack, sub_param.GetValue<size_t>(), __FUNCTION__));

        // If the discretization exists, the volumetric source can be initialized.
        if (discretization_)
          volumetric_sources_.back().Initialize(*this);
      }
    }

    else if (spec.Name() == "param_id")
      options_.param_id = spec.GetValue<int>();

    else if (spec.Name() == "phase")
      options_.phase = spec.GetValue<std::string>();
  } // for p

  if (options_.write_restart_time_interval > 0)
  {
    auto dir = options_.write_restart_path.parent_path();
    if (opensn::mpi_comm.rank() == 0)
      std::filesystem::create_directory(dir);
    opensn::mpi_comm.barrier();
    if (not std::filesystem::is_directory(dir))
      throw std::runtime_error("Failed to create restart directory " + dir.string());
  }
}

void
LBSSolver::SetBoundaryOptions(const InputParameters& params)
{
  const std::string fname = __FUNCTION__;
  const auto& user_params = params.ParametersAtAssignment();
  const auto boundary_name = user_params.GetParamValue<std::string>("name");
  const auto bndry_type = user_params.GetParamValue<std::string>("type");

  const auto bid = supported_boundary_names.at(boundary_name);
  const std::map<std::string, BoundaryType> type_list = {{"vacuum", BoundaryType::VACUUM},
                                                         {"isotropic", BoundaryType::ISOTROPIC},
                                                         {"reflecting", BoundaryType::REFLECTING},
                                                         {"arbitrary", BoundaryType::ARBITRARY}};

  const auto type = type_list.at(bndry_type);
  switch (type)
  {
    case BoundaryType::VACUUM:
    case BoundaryType::REFLECTING:
    {
      BoundaryPreferences()[bid] = {type};
      break;
    }
    case BoundaryType::ISOTROPIC:
    {
      OpenSnInvalidArgumentIf(not user_params.Has("group_strength"),
                              "Boundary conditions with type=\"isotropic\" require parameter "
                              "\"group_strength\"");

      user_params.RequireParameterBlockTypeIs("group_strength", ParameterBlockType::ARRAY);
      const auto group_strength = user_params.GetParamVectorValue<double>("group_strength");
      boundary_preferences_[bid] = {type, group_strength};
      break;
    }
    case BoundaryType::ARBITRARY:
    {
      throw std::runtime_error("Arbitrary boundary conditions are not currently supported.");
      break;
    }
  }
}

void
LBSSolver::Initialize()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::Initialize");

  PerformInputChecks(); // assigns num_groups and grid
  PrintSimHeader();

  mpi_comm.barrier();

  InitializeMaterials();
  InitializeSpatialDiscretization();
  InitializeGroupsets();
  ComputeNumberOfMoments();
  InitializeParrays();
  InitializeBoundaries();

  // Initialize point sources
  for (auto& point_source : point_sources_)
    point_source.Initialize(*this);

  // Initialize volumetric sources
  for (auto& volumetric_source : volumetric_sources_)
    volumetric_source.Initialize(*this);
}

void
LBSSolver::PerformInputChecks()
{
  if (groups_.empty())
  {
    log.LogAllError() << "LinearBoltzmann::SteadyStateSolver: No groups added to solver.";
    Exit(EXIT_FAILURE);
  }

  num_groups_ = groups_.size();

  if (groupsets_.empty())
  {
    log.LogAllError() << "LinearBoltzmann::SteadyStateSolver: No group-sets added to solver.";
    Exit(EXIT_FAILURE);
  }
  int grpset_counter = 0;
  for (auto& group_set : groupsets_)
  {
    if (group_set.groups.empty())
    {
      log.LogAllError() << "LinearBoltzmann::SteadyStateSolver: No groups added to groupset "
                        << grpset_counter << ".";
      Exit(EXIT_FAILURE);
    }
    ++grpset_counter;
  }

  if (options_.sd_type == SpatialDiscretizationType::UNDEFINED)
  {
    log.LogAllError() << "LinearBoltzmann::SteadyStateSolver: No discretization_ method set.";
    Exit(EXIT_FAILURE);
  }

  grid_ptr_ = GetCurrentMesh();

  if (grid_ptr_ == nullptr)
  {
    log.LogAllError() << "LinearBoltzmann::SteadyStateSolver: No "
                         "grid_ptr_ available from region.";
    Exit(EXIT_FAILURE);
  }

  // Determine geometry type
  const auto dim = grid_ptr_->Dimension();
  if (dim == 1)
    options_.geometry_type = GeometryType::ONED_SLAB;
  else if (dim == 2)
    options_.geometry_type = GeometryType::TWOD_CARTESIAN;
  else if (dim == 3)
    options_.geometry_type = GeometryType::THREED_CARTESIAN;
  else
    OpenSnLogicalError("Cannot deduce geometry type from mesh.");

  // Assign placeholder unit densities
  densities_local_.assign(grid_ptr_->local_cells.size(), 1.0);
}

void
LBSSolver::PrintSimHeader()
{
  if (opensn::mpi_comm.rank() == 0)
  {
    std::stringstream outstr;
    outstr << "\nInitializing LBS SteadyStateSolver with name: " << Name() << "\n\n"
           << "Scattering order    : " << options_.scattering_order << "\n"
           << "Number of Groups    : " << groups_.size() << "\n"
           << "Number of Group sets: " << groupsets_.size() << std::endl;

    // Output Groupsets
    for (const auto& groupset : groupsets_)
    {
      char buf_pol[20];

      outstr << "\n***** Groupset " << groupset.id << " *****\n"
             << "Groups:\n";
      int counter = 0;
      for (auto group : groupset.groups)
      {
        snprintf(buf_pol, 20, "%5d ", group.id);
        outstr << std::string(buf_pol);
        counter++;
        if (counter == 12)
        {
          counter = 0;
          outstr << "\n";
        }
      }
    }
    log.Log() << outstr.str() << "\n" << std::endl;
  }
}

void
LBSSolver::InitializeMaterials()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitializeMaterials");

  log.Log0Verbose1() << "Initializing Materials";

  // Create set of material ids locally relevant
  const size_t num_physics_mats = material_stack.size();

  int invalid_mat_cell_count = 0;
  std::set<int> unique_material_ids;
  for (auto& cell : grid_ptr_->local_cells)
  {
    unique_material_ids.insert(cell.material_id);
    if (cell.material_id < 0)
      ++invalid_mat_cell_count;
  }
  const auto& ghost_cell_ids = grid_ptr_->cells.GetGhostGlobalIDs();
  for (uint64_t cell_id : ghost_cell_ids)
  {
    const auto& cell = grid_ptr_->cells[cell_id];
    unique_material_ids.insert(cell.material_id);
    if (cell.material_id < 0)
      ++invalid_mat_cell_count;
  }
  OpenSnLogicalErrorIf(invalid_mat_cell_count > 0,
                       std::to_string(invalid_mat_cell_count) +
                         " cells encountered with an invalid material id.");

  // Get ready for processing
  std::stringstream materials_list;
  matid_to_xs_map_.clear();
  matid_to_src_map_.clear();

  // Process materials
  for (const int& mat_id : unique_material_ids)
  {
    materials_list << "Material id " << mat_id;

    const auto& current_material = GetStackItemPtr(material_stack, mat_id, __FUNCTION__);

    // Extract properties
    bool found_transport_xs = false;
    for (const auto& property : current_material->properties)
    {
      if (property->Type() == PropertyType::TRANSPORT_XSECTIONS)
      {
        auto xs = std::static_pointer_cast<MultiGroupXS>(property);
        xs->SetAdjointMode(options_.adjoint);
        matid_to_xs_map_[mat_id] = xs;
        found_transport_xs = true;
      }

      if (property->Type() == PropertyType::ISOTROPIC_MG_SOURCE)
      {
        const auto& src = std::static_pointer_cast<IsotropicMultiGroupSource>(property);

        // Check for a valid source
        if (src->source_value_g.size() < groups_.size())
        {
          log.LogAllWarning() << __FUNCTION__ << ": IsotropicMultiGroupSource specified in "
                              << "material \"" << current_material->name << "\" has fewer "
                              << "energy groups than called for in the simulation. "
                              << "Source will be ignored.";
        }

        // Set the source if in forward mode
        // Material sources are currently unused in adjoint mode
        if (not options_.adjoint)
          matid_to_src_map_[mat_id] = src;
      } // P0 source
    }   // for property

    // Check valid property
    OpenSnLogicalErrorIf(not found_transport_xs,
                         "Material \"" + current_material->name + "\" does not contain " +
                           "transport cross sections.");

    // Check number of groups legal
    OpenSnLogicalErrorIf(matid_to_xs_map_[mat_id]->NumGroups() < groups_.size(),
                         "Material \"" + current_material->name + "\" has fewer groups (" +
                           std::to_string(matid_to_xs_map_[mat_id]->NumGroups()) + ") than " +
                           "the simulation (" + std::to_string(groups_.size()) + "). " +
                           "A material must have at least as many groups as the simulation.");

    // Check number of moments
    if (matid_to_xs_map_[mat_id]->ScatteringOrder() < options_.scattering_order)
    {
      log.Log0Warning() << __FUNCTION__ << ": Material \"" << current_material->name
                        << "\" has a lower scattering order ("
                        << matid_to_xs_map_[mat_id]->ScatteringOrder() << ") "
                        << "than the simulation (" << options_.scattering_order << ").";
    }

    materials_list << " number of moments " << matid_to_xs_map_[mat_id]->ScatteringOrder() + 1
                   << "\n";
  } // for material id

  // Initialize precursor properties
  num_precursors_ = 0;
  max_precursors_per_material_ = 0;
  for (const auto& mat_id_xs : matid_to_xs_map_)
  {
    const auto& xs = mat_id_xs.second;
    num_precursors_ += xs->NumPrecursors();
    if (xs->NumPrecursors() > max_precursors_per_material_)
      max_precursors_per_material_ = xs->NumPrecursors();
  }

  // if no precursors, turn off precursors
  if (num_precursors_ == 0)
    options_.use_precursors = false;

  // check compatibility when precursors are on
  if (options_.use_precursors)
  {
    for (const auto& [mat_id, xs] : matid_to_xs_map_)
    {
      OpenSnLogicalErrorIf(xs->IsFissionable() and num_precursors_ == 0,
                           "Incompatible cross-section data encountered for material ID " +
                             std::to_string(mat_id) + ". When delayed neutron data is present " +
                             "for one fissionable matrial, it must be present for all fissionable "
                             "materials.");
    }
  }

  // Update transport views if available
  if (grid_ptr_->local_cells.size() == cell_transport_views_.size())
    for (const auto& cell : grid_ptr_->local_cells)
    {
      const auto& xs_ptr = matid_to_xs_map_[cell.material_id];
      auto& transport_view = cell_transport_views_[cell.local_id];

      transport_view.ReassignXS(*xs_ptr);
    }

  log.Log0Verbose1() << "Materials Initialized:\n" << materials_list.str() << "\n";

  mpi_comm.barrier();
}

void
LBSSolver::InitializeSpatialDiscretization()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitializeSpatialDiscretization");

  log.Log() << "Initializing spatial discretization.\n";
  discretization_ = PieceWiseLinearDiscontinuous::New(*grid_ptr_);

  ComputeUnitIntegrals();
}

void
LBSSolver::ComputeUnitIntegrals()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::ComputeUnitIntegrals");

  log.Log() << "Computing unit integrals.\n";
  const auto& sdm = *discretization_;

  // Define spatial weighting functions
  struct SpatialWeightFunction // SWF
  {
    virtual double operator()(const Vector3& pt) const { return 1.0; }
    virtual ~SpatialWeightFunction() = default;
  };

  struct SphericalSWF : public SpatialWeightFunction
  {
    double operator()(const Vector3& pt) const override { return pt[2] * pt[2]; }
  };

  struct CylindricalSWF : public SpatialWeightFunction
  {
    double operator()(const Vector3& pt) const override { return pt[0]; }
  };

  auto swf_ptr = std::make_shared<SpatialWeightFunction>();
  if (options_.geometry_type == GeometryType::ONED_SPHERICAL)
    swf_ptr = std::make_shared<SphericalSWF>();
  if (options_.geometry_type == GeometryType::TWOD_CYLINDRICAL)
    swf_ptr = std::make_shared<CylindricalSWF>();

  auto ComputeCellUnitIntegrals = [&sdm](const Cell& cell, const SpatialWeightFunction& swf)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_faces = cell.faces.size();
    const size_t cell_num_nodes = cell_mapping.NumNodes();
    const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();

    DenseMatrix<double> IntV_gradshapeI_gradshapeJ(cell_num_nodes, cell_num_nodes, 0.0);
    DenseMatrix<Vector3> IntV_shapeI_gradshapeJ(cell_num_nodes, cell_num_nodes);
    DenseMatrix<double> IntV_shapeI_shapeJ(cell_num_nodes, cell_num_nodes, 0.0);
    Vector<double> IntV_shapeI(cell_num_nodes, 0.);
    std::vector<DenseMatrix<double>> IntS_shapeI_shapeJ(cell_num_faces);
    std::vector<DenseMatrix<Vector3>> IntS_shapeI_gradshapeJ(cell_num_faces);
    std::vector<Vector<double>> IntS_shapeI(cell_num_faces);

    // Volume integrals
    for (unsigned int i = 0; i < cell_num_nodes; ++i)
    {
      for (unsigned int j = 0; j < cell_num_nodes; ++j)
      {
        for (const auto& qp : fe_vol_data.QuadraturePointIndices())
        {
          IntV_gradshapeI_gradshapeJ(i, j) +=
            swf(fe_vol_data.QPointXYZ(qp)) *
            fe_vol_data.ShapeGrad(i, qp).Dot(fe_vol_data.ShapeGrad(j, qp)) *
            fe_vol_data.JxW(qp); // K-matrix

          IntV_shapeI_gradshapeJ(i, j) +=
            swf(fe_vol_data.QPointXYZ(qp)) * fe_vol_data.ShapeValue(i, qp) *
            fe_vol_data.ShapeGrad(j, qp) * fe_vol_data.JxW(qp); // G-matrix

          IntV_shapeI_shapeJ(i, j) +=
            swf(fe_vol_data.QPointXYZ(qp)) * fe_vol_data.ShapeValue(i, qp) *
            fe_vol_data.ShapeValue(j, qp) * fe_vol_data.JxW(qp); // M-matrix
        }                                                        // for qp
      }                                                          // for j

      for (const auto& qp : fe_vol_data.QuadraturePointIndices())
      {
        IntV_shapeI(i) +=
          swf(fe_vol_data.QPointXYZ(qp)) * fe_vol_data.ShapeValue(i, qp) * fe_vol_data.JxW(qp);
      } // for qp
    }   // for i

    //  surface integrals
    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
      IntS_shapeI_shapeJ[f] = DenseMatrix<double>(cell_num_nodes, cell_num_nodes, 0.0);
      IntS_shapeI[f] = Vector<double>(cell_num_nodes, 0.);
      IntS_shapeI_gradshapeJ[f] = DenseMatrix<Vector3>(cell_num_nodes, cell_num_nodes);

      for (unsigned int i = 0; i < cell_num_nodes; ++i)
      {
        for (unsigned int j = 0; j < cell_num_nodes; ++j)
        {
          for (const auto& qp : fe_srf_data.QuadraturePointIndices())
          {
            IntS_shapeI_shapeJ[f](i, j) += swf(fe_srf_data.QPointXYZ(qp)) *
                                           fe_srf_data.ShapeValue(i, qp) *
                                           fe_srf_data.ShapeValue(j, qp) * fe_srf_data.JxW(qp);
            IntS_shapeI_gradshapeJ[f](i, j) += swf(fe_srf_data.QPointXYZ(qp)) *
                                               fe_srf_data.ShapeValue(i, qp) *
                                               fe_srf_data.ShapeGrad(j, qp) * fe_srf_data.JxW(qp);
          } // for qp
        }   // for j

        for (const auto& qp : fe_srf_data.QuadraturePointIndices())
        {
          IntS_shapeI[f](i) +=
            swf(fe_srf_data.QPointXYZ(qp)) * fe_srf_data.ShapeValue(i, qp) * fe_srf_data.JxW(qp);
        } // for qp
      }   // for i
    }     // for f

    return UnitCellMatrices{IntV_gradshapeI_gradshapeJ,
                            IntV_shapeI_gradshapeJ,
                            IntV_shapeI_shapeJ,
                            IntV_shapeI,

                            IntS_shapeI_shapeJ,
                            IntS_shapeI_gradshapeJ,
                            IntS_shapeI};
  };

  const size_t num_local_cells = grid_ptr_->local_cells.size();
  unit_cell_matrices_.resize(num_local_cells);

  for (const auto& cell : grid_ptr_->local_cells)
    unit_cell_matrices_[cell.local_id] = ComputeCellUnitIntegrals(cell, *swf_ptr);

  const auto ghost_ids = grid_ptr_->cells.GetGhostGlobalIDs();
  for (uint64_t ghost_id : ghost_ids)
    unit_ghost_cell_matrices_[ghost_id] =
      ComputeCellUnitIntegrals(grid_ptr_->cells[ghost_id], *swf_ptr);

  // Assessing global unit cell matrix storage
  std::array<size_t, 2> num_local_ucms = {unit_cell_matrices_.size(),
                                          unit_ghost_cell_matrices_.size()};
  std::array<size_t, 2> num_globl_ucms = {0, 0};

  mpi_comm.all_reduce(num_local_ucms.data(), 2, num_globl_ucms.data(), mpi::op::sum<size_t>());

  opensn::mpi_comm.barrier();
  log.Log() << "Ghost cell unit cell-matrix ratio: "
            << (double)num_globl_ucms[1] * 100 / (double)num_globl_ucms[0] << "%";
  log.Log() << "Cell matrices computed.";
}

void
LBSSolver::InitializeGroupsets()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitializeGroupsets");

  for (auto& groupset : groupsets_)
  {
    // Build groupset angular flux unknown manager
    groupset.psi_uk_man_.unknowns.clear();
    size_t num_angles = groupset.quadrature->abscissae.size();
    size_t gs_num_groups = groupset.groups.size();
    auto& grpset_psi_uk_man = groupset.psi_uk_man_;

    const auto VarVecN = UnknownType::VECTOR_N;
    for (unsigned int n = 0; n < num_angles; ++n)
      grpset_psi_uk_man.AddUnknown(VarVecN, gs_num_groups);

    groupset.BuildDiscMomOperator(options_.scattering_order, options_.geometry_type);
    groupset.BuildMomDiscOperator(options_.scattering_order, options_.geometry_type);
    groupset.BuildSubsets();
  } // for groupset
}

void
LBSSolver::ComputeNumberOfMoments()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::ComputeNumberOfMoments");

  for (size_t gs = 1; gs < groupsets_.size(); ++gs)
    if (groupsets_[gs].quadrature->GetMomentToHarmonicsIndexMap() !=
        groupsets_[0].quadrature->GetMomentToHarmonicsIndexMap())
      throw std::logic_error("LinearBoltzmann::SteadyStateSolver::ComputeNumberOfMoments : "
                             "Moment-to-Harmonics mapping differs between "
                             "groupsets_, which is not allowed.");

  num_moments_ = (int)groupsets_.front().quadrature->GetMomentToHarmonicsIndexMap().size();

  if (num_moments_ == 0)
    throw std::logic_error("LinearBoltzmann::SteadyStateSolver::ComputeNumberOfMoments : "
                           "unable to infer number of moments from angular "
                           "quadrature.");
}

void
LBSSolver::InitializeParrays()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitializeParrays");

  log.Log() << "Initializing parallel arrays."
            << " G=" << num_groups_ << " M=" << num_moments_ << std::endl;

  // Initialize unknown
  // structure
  flux_moments_uk_man_.unknowns.clear();
  for (size_t m = 0; m < num_moments_; ++m)
  {
    flux_moments_uk_man_.AddUnknown(UnknownType::VECTOR_N, groups_.size());
    flux_moments_uk_man_.unknowns.back().name = "m" + std::to_string(m);
  }

  // Compute local # of dof
  auto per_node = UnknownManager::GetUnitaryUnknownManager();
  local_node_count_ = discretization_->GetNumLocalDOFs(per_node);
  glob_node_count_ = discretization_->GetNumGlobalDOFs(per_node);

  // Compute num of unknowns
  size_t num_grps = groups_.size();
  size_t local_unknown_count = local_node_count_ * num_grps * num_moments_;

  log.LogAllVerbose1() << "LBS Number of phi unknowns: " << local_unknown_count;

  // Size local vectors
  q_moments_local_.assign(local_unknown_count, 0.0);
  phi_old_local_.assign(local_unknown_count, 0.0);
  phi_new_local_.assign(local_unknown_count, 0.0);

  // Setup groupset psi vectors
  psi_new_local_.clear();
  for (auto& groupset : groupsets_)
  {
    psi_new_local_.emplace_back();
    if (options_.save_angular_flux)
    {
      size_t num_ang_unknowns = discretization_->GetNumLocalDOFs(groupset.psi_uk_man_);
      psi_new_local_.back().assign(num_ang_unknowns, 0.0);
    }
  }

  // Setup precursor vector
  if (options_.use_precursors)
  {
    size_t num_precursor_dofs = grid_ptr_->local_cells.size() * max_precursors_per_material_;
    precursor_new_local_.assign(num_precursor_dofs, 0.0);
  }

  // Read Restart data
  if (not options_.read_restart_path.empty())
    ReadRestartData();
  opensn::mpi_comm.barrier();

  // Initialize transport views
  // Transport views act as a data structure to store information
  // related to the transport simulation. The most prominent function
  // here is that it holds the means to know where a given cell's
  // transport quantities are located in the unknown vectors (i.e. phi)
  //
  // Also, for a given cell, within a given sweep chunk,
  // we need to solve a matrix which square size is the
  // amount of nodes on the cell. max_cell_dof_count is
  // initialized here.
  //
  size_t block_MG_counter = 0; // Counts the strides of moment and group

  const Vector3 ihat(1.0, 0.0, 0.0);
  const Vector3 jhat(0.0, 1.0, 0.0);
  const Vector3 khat(0.0, 0.0, 1.0);

  max_cell_dof_count_ = 0;
  cell_transport_views_.clear();
  cell_transport_views_.reserve(grid_ptr_->local_cells.size());
  for (auto& cell : grid_ptr_->local_cells)
  {
    size_t num_nodes = discretization_->GetCellNumNodes(cell);
    int mat_id = cell.material_id;

    // compute cell volumes
    double cell_volume = 0.0;
    const auto& IntV_shapeI = unit_cell_matrices_[cell.local_id].intV_shapeI;
    for (size_t i = 0; i < num_nodes; ++i)
      cell_volume += IntV_shapeI(i);

    size_t cell_phi_address = block_MG_counter;

    const size_t num_faces = cell.faces.size();
    std::vector<bool> face_local_flags(num_faces, true);
    std::vector<int> face_locality(num_faces, opensn::mpi_comm.rank());
    std::vector<const Cell*> neighbor_cell_ptrs(num_faces, nullptr);
    bool cell_on_boundary = false;
    int f = 0;
    for (auto& face : cell.faces)
    {
      if (not face.has_neighbor)
      {
        Vector3& n = face.normal;

        int boundary_id = -1;
        if (n.Dot(ihat) < -0.999)
          boundary_id = XMIN;
        else if (n.Dot(ihat) > 0.999)
          boundary_id = XMAX;
        else if (n.Dot(jhat) < -0.999)
          boundary_id = YMIN;
        else if (n.Dot(jhat) > 0.999)
          boundary_id = YMAX;
        else if (n.Dot(khat) < -0.999)
          boundary_id = ZMIN;
        else if (n.Dot(khat) > 0.999)
          boundary_id = ZMAX;

        if (boundary_id >= 0)
          face.neighbor_id = boundary_id;
        cell_on_boundary = true;

        face_local_flags[f] = false;
        face_locality[f] = -1;
      } // if bndry
      else
      {
        const int neighbor_partition = face.GetNeighborPartitionID(*grid_ptr_);
        face_local_flags[f] = (neighbor_partition == opensn::mpi_comm.rank());
        face_locality[f] = neighbor_partition;
        neighbor_cell_ptrs[f] = &grid_ptr_->cells[face.neighbor_id];
      }

      ++f;
    } // for f

    if (num_nodes > max_cell_dof_count_)
      max_cell_dof_count_ = num_nodes;

    cell_transport_views_.emplace_back(cell_phi_address,
                                       num_nodes,
                                       num_grps,
                                       num_moments_,
                                       num_faces,
                                       *matid_to_xs_map_[mat_id],
                                       cell_volume,
                                       face_local_flags,
                                       face_locality,
                                       neighbor_cell_ptrs,
                                       cell_on_boundary);
    block_MG_counter += num_nodes * num_grps * num_moments_;
  } // for local cell

  // Populate grid nodal mappings
  // This is used in the Flux Data Structures (FLUDS)
  grid_nodal_mappings_.clear();
  grid_nodal_mappings_.reserve(grid_ptr_->local_cells.size());
  for (auto& cell : grid_ptr_->local_cells)
  {
    CellFaceNodalMapping cell_nodal_mapping;
    cell_nodal_mapping.reserve(cell.faces.size());

    for (auto& face : cell.faces)
    {
      std::vector<short> face_node_mapping;
      std::vector<short> cell_node_mapping;
      int ass_face = -1;

      if (face.has_neighbor)
      {
        grid_ptr_->FindAssociatedVertices(face, face_node_mapping);
        grid_ptr_->FindAssociatedCellVertices(face, cell_node_mapping);
        ass_face = face.GetNeighborAssociatedFace(*grid_ptr_);
      }

      cell_nodal_mapping.emplace_back(ass_face, face_node_mapping, cell_node_mapping);
    } // for f

    grid_nodal_mappings_.push_back(cell_nodal_mapping);
  } // for local cell

  // Get grid localized communicator set
  grid_local_comm_set_ = grid_ptr_->MakeMPILocalCommunicatorSet();

  // Make face histogram
  grid_face_histogram_ = grid_ptr_->MakeGridFaceHistogram();

  // Initialize Field Functions
  InitializeFieldFunctions();

  opensn::mpi_comm.barrier();
  log.Log() << "Done with parallel arrays." << std::endl;
}

void
LBSSolver::InitializeFieldFunctions()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitializeFieldFunctions");

  if (not field_functions_.empty())
    return;

  // Initialize Field Functions
  //                                              for flux moments
  phi_field_functions_local_map_.clear();

  for (size_t g = 0; g < groups_.size(); ++g)
  {
    for (size_t m = 0; m < num_moments_; ++m)
    {
      std::string prefix;
      if (options_.field_function_prefix_option == "prefix")
      {
        prefix = options_.field_function_prefix;
        if (not prefix.empty())
          prefix += "_";
      }
      if (options_.field_function_prefix_option == "solver_name")
        prefix = Name() + "_";

      char buff[100];
      snprintf(
        buff, 99, "%sphi_g%03d_m%02d", prefix.c_str(), static_cast<int>(g), static_cast<int>(m));
      const std::string name = std::string(buff);

      auto group_ff = std::make_shared<FieldFunctionGridBased>(
        name, discretization_, Unknown(UnknownType::SCALAR));

      field_function_stack.push_back(group_ff);
      field_functions_.push_back(group_ff);

      phi_field_functions_local_map_[{g, m}] = field_functions_.size() - 1;
    } // for m
  }   // for g

  // Initialize power generation field function
  if (options_.power_field_function_on)
  {
    std::string prefix;
    if (options_.field_function_prefix_option == "prefix")
    {
      prefix = options_.field_function_prefix;
      if (not prefix.empty())
        prefix += "_";
    }
    if (options_.field_function_prefix_option == "solver_name")
      prefix = Name() + "_";

    auto power_ff = std::make_shared<FieldFunctionGridBased>(
      prefix + "power_generation", discretization_, Unknown(UnknownType::SCALAR));

    field_function_stack.push_back(power_ff);
    field_functions_.push_back(power_ff);

    power_gen_fieldfunc_local_handle_ = field_functions_.size() - 1;
  }
}

void
LBSSolver::InitializeBoundaries()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitializeBoundaries");

  const std::string fname = "LBSSolver::InitializeBoundaries";
  // Determine boundary-ids involved in the problem
  std::set<uint64_t> globl_unique_bids_set;
  {
    std::set<uint64_t> local_unique_bids_set;
    for (const auto& cell : grid_ptr_->local_cells)
      for (const auto& face : cell.faces)
        if (not face.has_neighbor)
          local_unique_bids_set.insert(face.neighbor_id);

    std::vector<uint64_t> local_unique_bids(local_unique_bids_set.begin(),
                                            local_unique_bids_set.end());
    std::vector<uint64_t> recvbuf;
    mpi_comm.all_gather(local_unique_bids, recvbuf);

    globl_unique_bids_set = local_unique_bids_set; // give it a head start

    for (uint64_t bid : recvbuf)
      globl_unique_bids_set.insert(bid);
  }

  // Initialize default incident boundary
  const size_t G = num_groups_;

  sweep_boundaries_.clear();
  for (uint64_t bid : globl_unique_bids_set)
  {
    const bool has_no_preference = boundary_preferences_.count(bid) == 0;
    const bool has_not_been_set = sweep_boundaries_.count(bid) == 0;
    if (has_no_preference and has_not_been_set)
    {
      sweep_boundaries_[bid] = std::make_shared<VacuumBoundary>(G);
    } // defaulted
    else if (has_not_been_set)
    {
      const auto& bndry_pref = boundary_preferences_.at(bid);
      const auto& mg_q = bndry_pref.isotropic_mg_source;

      if (bndry_pref.type == BoundaryType::VACUUM)
        sweep_boundaries_[bid] = std::make_shared<VacuumBoundary>(G);
      else if (bndry_pref.type == BoundaryType::ISOTROPIC)
        sweep_boundaries_[bid] = std::make_shared<IsotropicBoundary>(G, mg_q);
      else if (bndry_pref.type == BoundaryType::REFLECTING)
      {
        // Locally check all faces, that subscribe to this boundary,
        // have the same normal
        const double EPSILON = 1.0e-12;
        std::unique_ptr<Vector3> n_ptr = nullptr;
        for (const auto& cell : grid_ptr_->local_cells)
          for (const auto& face : cell.faces)
            if (not face.has_neighbor and face.neighbor_id == bid)
            {
              if (not n_ptr)
                n_ptr = std::make_unique<Vector3>(face.normal);
              if (std::fabs(face.normal.Dot(*n_ptr) - 1.0) > EPSILON)
                throw std::logic_error(fname +
                                       ": Not all face normals are, within tolerance, locally the "
                                       "same for the reflecting boundary condition requested.");
            }

        // Now check globally
        const int local_has_bid = n_ptr != nullptr ? 1 : 0;
        const Vector3 local_normal = local_has_bid ? *n_ptr : Vector3(0.0, 0.0, 0.0);

        std::vector<int> locJ_has_bid(opensn::mpi_comm.size(), 1);
        std::vector<double> locJ_n_val(opensn::mpi_comm.size() * 3, 0.0);

        mpi_comm.all_gather(local_has_bid, locJ_has_bid);
        std::vector<double> lnv = {local_normal.x, local_normal.y, local_normal.z};
        mpi_comm.all_gather(lnv.data(), 3, locJ_n_val.data(), 3);

        Vector3 global_normal;
        for (int j = 0; j < opensn::mpi_comm.size(); ++j)
        {
          if (locJ_has_bid[j])
          {
            int offset = 3 * j;
            const double* n = &locJ_n_val[offset];
            const Vector3 locJ_normal(n[0], n[1], n[2]);

            if (local_has_bid)
              if (std::fabs(local_normal.Dot(locJ_normal) - 1.0) > EPSILON)
                throw std::logic_error(fname +
                                       ": Not all face normals are, within tolerance, globally the "
                                       "same for the reflecting boundary condition requested.");

            global_normal = locJ_normal;
          }
        }

        sweep_boundaries_[bid] = std::make_shared<ReflectingBoundary>(
          G, global_normal, MapGeometryTypeToCoordSys(options_.geometry_type));
      }
    } // non-defaulted
  }   // for bndry id
}

void
LBSSolver::InitializeSolverSchemes()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitializeSolverSchemes");

  log.Log() << "Initializing WGS and AGS solvers";

  InitializeWGSSolvers();

  ags_solver_ = std::make_shared<AGSSolver>(*this, wgs_solvers_);
  if (groupsets_.size() == 1)
  {
    ags_solver_->MaxIterations(1);
    ags_solver_->Verbosity(false);
  }
  else
  {
    ags_solver_->MaxIterations(options_.max_ags_iterations);
    ags_solver_->Verbosity(options_.verbose_ags_iterations);
  }
  ags_solver_->Tolerance(options_.ags_tolerance);
}

void
LBSSolver::InitWGDSA(LBSGroupset& groupset, bool vaccum_bcs_are_dirichlet)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitWGDSA");

  if (groupset.apply_wgdsa)
  {
    // Make UnknownManager
    const size_t num_gs_groups = groupset.groups.size();
    opensn::UnknownManager uk_man;
    uk_man.AddUnknown(UnknownType::VECTOR_N, num_gs_groups);

    // Make boundary conditions
    auto bcs = TranslateBCs(sweep_boundaries_, vaccum_bcs_are_dirichlet);

    // Make xs map
    auto matid_2_mgxs_map =
      PackGroupsetXS(matid_to_xs_map_, groupset.groups.front().id, groupset.groups.back().id);

    // Create solver
    const auto& sdm = *discretization_;

    auto solver = std::make_shared<DiffusionMIPSolver>(std::string(Name() + "_WGDSA"),
                                                       sdm,
                                                       uk_man,
                                                       bcs,
                                                       matid_2_mgxs_map,
                                                       unit_cell_matrices_,
                                                       false,
                                                       true);
    ParameterBlock block;

    solver->options.residual_tolerance = groupset.wgdsa_tol;
    solver->options.max_iters = groupset.wgdsa_max_iters;
    solver->options.verbose = groupset.wgdsa_verbose;
    solver->options.additional_options_string = groupset.wgdsa_string;

    solver->Initialize();

    std::vector<double> dummy_rhs(sdm.GetNumLocalDOFs(uk_man), 0.0);

    solver->AssembleAand_b(dummy_rhs);

    groupset.wgdsa_solver = solver;
  }
}

void
LBSSolver::CleanUpWGDSA(LBSGroupset& groupset)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::CleanUpWGDSA");

  if (groupset.apply_wgdsa)
    groupset.wgdsa_solver = nullptr;
}

std::vector<double>
LBSSolver::WGSCopyOnlyPhi0(const LBSGroupset& groupset, const std::vector<double>& phi_in)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::WGSCopyOnlyPhi0");

  const auto& sdm = *discretization_;
  const auto& dphi_uk_man = groupset.wgdsa_solver->UnknownStructure();
  const auto& phi_uk_man = flux_moments_uk_man_;

  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();

  std::vector<double> output_phi_local(sdm.GetNumLocalDOFs(dphi_uk_man), 0.0);

  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t dphi_map = sdm.MapDOFLocal(cell, i, dphi_uk_man, 0, 0);
      const int64_t phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, gsi);

      double* output_mapped = &output_phi_local[dphi_map];
      const double* phi_in_mapped = &phi_in[phi_map];

      for (size_t g = 0; g < gss; ++g)
      {
        output_mapped[g] = phi_in_mapped[g];
      } // for g
    }   // for node
  }     // for cell

  return output_phi_local;
}

void
LBSSolver::GSProjectBackPhi0(const LBSGroupset& groupset,
                             const std::vector<double>& input,
                             std::vector<double>& output)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::GSProjectBackPhi0");

  const auto& sdm = *discretization_;
  const auto& dphi_uk_man = groupset.wgdsa_solver->UnknownStructure();
  const auto& phi_uk_man = flux_moments_uk_man_;

  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();

  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t dphi_map = sdm.MapDOFLocal(cell, i, dphi_uk_man, 0, 0);
      const int64_t phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, gsi);

      const double* input_mapped = &input[dphi_map];
      double* output_mapped = &output[phi_map];

      for (int g = 0; g < gss; ++g)
        output_mapped[g] = input_mapped[g];
    } // for dof
  }   // for cell
}

void
LBSSolver::AssembleWGDSADeltaPhiVector(const LBSGroupset& groupset,
                                       const std::vector<double>& phi_in,
                                       std::vector<double>& delta_phi_local)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::AssembleWGDSADeltaPhiVector");

  const auto& sdm = *discretization_;
  const auto& dphi_uk_man = groupset.wgdsa_solver->UnknownStructure();
  const auto& phi_uk_man = flux_moments_uk_man_;

  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();

  delta_phi_local.clear();
  delta_phi_local.assign(sdm.GetNumLocalDOFs(dphi_uk_man), 0.0);

  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();
    const auto& sigma_s = matid_to_xs_map_[cell.material_id]->SigmaSGtoG();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t dphi_map = sdm.MapDOFLocal(cell, i, dphi_uk_man, 0, 0);
      const int64_t phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, gsi);

      double* delta_phi_mapped = &delta_phi_local[dphi_map];
      const double* phi_in_mapped = &phi_in[phi_map];

      for (size_t g = 0; g < gss; ++g)
      {
        delta_phi_mapped[g] = sigma_s[gsi + g] * phi_in_mapped[g];
      } // for g
    }   // for node
  }     // for cell
}

void
LBSSolver::DisAssembleWGDSADeltaPhiVector(const LBSGroupset& groupset,
                                          const std::vector<double>& delta_phi_local,
                                          std::vector<double>& ref_phi_new)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::DisAssembleWGDSADeltaPhiVector");

  const auto& sdm = *discretization_;
  const auto& dphi_uk_man = groupset.wgdsa_solver->UnknownStructure();
  const auto& phi_uk_man = flux_moments_uk_man_;

  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();

  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t dphi_map = sdm.MapDOFLocal(cell, i, dphi_uk_man, 0, 0);
      const int64_t phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, gsi);

      const double* delta_phi_mapped = &delta_phi_local[dphi_map];
      double* phi_new_mapped = &ref_phi_new[phi_map];

      for (int g = 0; g < gss; ++g)
        phi_new_mapped[g] += delta_phi_mapped[g];
    } // for dof
  }   // for cell
}

void
LBSSolver::InitTGDSA(LBSGroupset& groupset)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::InitTGDSA");

  if (groupset.apply_tgdsa)
  {
    // Make UnknownManager
    const auto& uk_man = discretization_->UNITARY_UNKNOWN_MANAGER;

    // Make boundary conditions
    auto bcs = TranslateBCs(sweep_boundaries_);

    // Make TwoGridInfo
    for (const auto& mat_id_xs_pair : matid_to_xs_map_)
    {
      const auto& mat_id = mat_id_xs_pair.first;
      const auto& xs = mat_id_xs_pair.second;

      TwoGridCollapsedInfo tginfo = MakeTwoGridCollapsedInfo(*xs, EnergyCollapseScheme::JFULL);

      groupset.tg_acceleration_info_.map_mat_id_2_tginfo.insert(
        std::make_pair(mat_id, std::move(tginfo)));
    }

    // Make xs map
    std::map<int, Multigroup_D_and_sigR> matid_2_mgxs_map;
    for (const auto& matid_xs_pair : matid_to_xs_map_)
    {
      const auto& mat_id = matid_xs_pair.first;

      const auto& tg_info = groupset.tg_acceleration_info_.map_mat_id_2_tginfo.at(mat_id);

      matid_2_mgxs_map.insert(std::make_pair(
        mat_id, Multigroup_D_and_sigR{{tg_info.collapsed_D}, {tg_info.collapsed_sig_a}}));
    }

    // Create solver
    const auto& sdm = *discretization_;

    auto solver = std::make_shared<DiffusionMIPSolver>(std::string(Name() + "_TGDSA"),
                                                       sdm,
                                                       uk_man,
                                                       bcs,
                                                       matid_2_mgxs_map,
                                                       unit_cell_matrices_,
                                                       false,
                                                       true);

    solver->options.residual_tolerance = groupset.tgdsa_tol;
    solver->options.max_iters = groupset.tgdsa_max_iters;
    solver->options.verbose = groupset.tgdsa_verbose;
    solver->options.additional_options_string = groupset.tgdsa_string;

    solver->Initialize();

    std::vector<double> dummy_rhs(sdm.GetNumLocalDOFs(uk_man), 0.0);

    solver->AssembleAand_b(dummy_rhs);

    groupset.tgdsa_solver = solver;
  }
}

void
LBSSolver::CleanUpTGDSA(LBSGroupset& groupset)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::CleanUpTGDSA");

  if (groupset.apply_tgdsa)
    groupset.tgdsa_solver = nullptr;
}

void
LBSSolver::AssembleTGDSADeltaPhiVector(const LBSGroupset& groupset,
                                       const std::vector<double>& phi_in,
                                       std::vector<double>& delta_phi_local)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::AssembleTGDSADeltaPhiVector");

  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;

  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();

  delta_phi_local.clear();
  delta_phi_local.assign(local_node_count_, 0.0);

  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();
    const auto& S = matid_to_xs_map_[cell.material_id]->TransferMatrix(0);

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t dphi_map = sdm.MapDOFLocal(cell, i);
      const int64_t phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, 0);

      double& delta_phi_mapped = delta_phi_local[dphi_map];
      const double* phi_in_mapped = &phi_in[phi_map];

      for (size_t g = 0; g < gss; ++g)
      {
        double R_g = 0.0;
        for (const auto& [row_g, gprime, sigma_sm] : S.Row(gsi + g))
          if (gprime >= gsi and gprime != (gsi + g))
            R_g += sigma_sm * phi_in_mapped[gprime];

        delta_phi_mapped += R_g;
      } // for g
    }   // for node
  }     // for cell
}

void
LBSSolver::DisAssembleTGDSADeltaPhiVector(const LBSGroupset& groupset,
                                          const std::vector<double>& delta_phi_local,
                                          std::vector<double>& ref_phi_new)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::DisAssembleTGDSADeltaPhiVector");

  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;

  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();

  const auto& map_mat_id_2_tginfo = groupset.tg_acceleration_info_.map_mat_id_2_tginfo;

  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    const auto& xi_g = map_mat_id_2_tginfo.at(cell.material_id).spectrum;

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t dphi_map = sdm.MapDOFLocal(cell, i);
      const int64_t phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, gsi);

      const double delta_phi_mapped = delta_phi_local[dphi_map];
      double* phi_new_mapped = &ref_phi_new[phi_map];

      for (int g = 0; g < gss; ++g)
        phi_new_mapped[g] += delta_phi_mapped * xi_g[gsi + g];
    } // for dof
  }   // for cell
}

bool
LBSSolver::TriggerRestartDump()
{
  auto now = std::chrono::system_clock::now().time_since_epoch();
  size_t now_secs = std::chrono::duration_cast<std::chrono::seconds>(now).count();

  if ((now_secs - last_restart_write_time_) < options_.write_restart_time_interval)
    return false;

  return true;
}

void
LBSSolver::UpdateLastRestartWriteTime()
{
  auto now = std::chrono::system_clock::now().time_since_epoch();
  last_restart_write_time_ = std::chrono::duration_cast<std::chrono::seconds>(now).count();
}

void
LBSSolver::WriteRestartData()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::WriteRestartData");

  std::string fbase = options_.write_restart_path.string();
  std::string fname = fbase + std::to_string(opensn::mpi_comm.rank()) + ".restart.h5";

  // Write data
  bool location_succeeded = true;
  auto file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file)
  {
    location_succeeded = H5WriteDataset1D<double>(file, "phi_old", phi_old_local_);
    H5Fclose(file);
  }
  else
    location_succeeded = false;

  bool global_succeeded = true;
  mpi_comm.all_reduce(location_succeeded, global_succeeded, mpi::op::logical_and<bool>());
  if (global_succeeded)
  {
    log.Log() << "Successfully wrote restart data to " << fbase << "X.restart.h5";
    UpdateLastRestartWriteTime();
  }
  else
    log.Log0Error() << "Failed to write restart data to " << fbase << "X.restart.h5";
}

void
LBSSolver::ReadRestartData()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::ReadRestartData");

  std::string fbase = options_.read_restart_path.string();
  std::string fname = fbase + std::to_string(opensn::mpi_comm.rank()) + ".restart.h5";

  bool location_succeeded = true;
  auto file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file)
  {
    phi_old_local_.clear();
    phi_old_local_ = H5ReadDataset1D<double>(file, "phi_old");
    location_succeeded = not phi_old_local_.empty();
    H5Fclose(file);
  }
  else
    location_succeeded = false;

  bool global_succeeded = true;
  mpi_comm.all_reduce(location_succeeded, global_succeeded, mpi::op::logical_and<bool>());
  if (global_succeeded)
    log.Log() << "Successfully read restart data from " << fbase + "X.restart.h5";
  else
    throw std::logic_error("Failed to read restart data from " + fbase + "X.restart.h5");
}

std::vector<double>
LBSSolver::MakeSourceMomentsFromPhi()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::MakeSourceMomentsFromPhi");

  size_t num_local_dofs = discretization_->GetNumLocalDOFs(flux_moments_uk_man_);

  std::vector<double> source_moments(num_local_dofs, 0.0);
  for (auto& groupset : groupsets_)
  {
    active_set_source_function_(groupset,
                                source_moments,
                                phi_new_local_,
                                APPLY_AGS_SCATTER_SOURCES | APPLY_WGS_SCATTER_SOURCES |
                                  APPLY_AGS_FISSION_SOURCES | APPLY_WGS_FISSION_SOURCES);
  }

  return source_moments;
}

void
LBSSolver::UpdateFieldFunctions()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::UpdateFieldFunctions");

  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;

  // Update flux moments
  for (const auto& [g_and_m, ff_index] : phi_field_functions_local_map_)
  {
    const size_t g = g_and_m.first;
    const size_t m = g_and_m.second;

    std::vector<double> data_vector_local(local_node_count_, 0.0);

    for (const auto& cell : grid_ptr_->local_cells)
    {
      const auto& cell_mapping = sdm.GetCellMapping(cell);
      const size_t num_nodes = cell_mapping.NumNodes();

      for (size_t i = 0; i < num_nodes; ++i)
      {
        const int64_t imapA = sdm.MapDOFLocal(cell, i, phi_uk_man, m, g);
        const int64_t imapB = sdm.MapDOFLocal(cell, i);

        data_vector_local[imapB] = phi_new_local_[imapA];
      } // for node
    }   // for cell

    auto& ff_ptr = field_functions_.at(ff_index);
    ff_ptr->UpdateFieldVector(data_vector_local);
  }

  // Update power generation
  if (options_.power_field_function_on)
  {
    std::vector<double> data_vector_local(local_node_count_, 0.0);

    double local_total_power = 0.0;
    for (const auto& cell : grid_ptr_->local_cells)
    {
      const auto& cell_mapping = sdm.GetCellMapping(cell);
      const size_t num_nodes = cell_mapping.NumNodes();

      const auto& Vi = unit_cell_matrices_[cell.local_id].intV_shapeI;

      const auto& xs = matid_to_xs_map_.at(cell.material_id);

      if (not xs->IsFissionable())
        continue;

      for (size_t i = 0; i < num_nodes; ++i)
      {
        const int64_t imapA = sdm.MapDOFLocal(cell, i);
        const int64_t imapB = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, 0);

        double nodal_power = 0.0;
        for (size_t g = 0; g < groups_.size(); ++g)
        {
          const double sigma_fg = xs->SigmaFission()[g];
          // const double kappa_g = xs->Kappa()[g];
          const double kappa_g = options_.power_default_kappa;

          nodal_power += kappa_g * sigma_fg * phi_new_local_[imapB + g];
        } // for g

        data_vector_local[imapA] = nodal_power;
        local_total_power += nodal_power * Vi(i);
      } // for node
    }   // for cell

    if (options_.power_normalization > 0.0)
    {
      double globl_total_power;
      mpi_comm.all_reduce(local_total_power, globl_total_power, mpi::op::sum<double>());

      Scale(data_vector_local, options_.power_normalization / globl_total_power);
    }

    const size_t ff_index = power_gen_fieldfunc_local_handle_;

    auto& ff_ptr = field_functions_.at(ff_index);
    ff_ptr->UpdateFieldVector(data_vector_local);

  } // if power enabled
}

void
LBSSolver::SetPhiFromFieldFunctions(PhiSTLOption which_phi,
                                    const std::vector<size_t>& m_indices,
                                    const std::vector<size_t>& g_indices)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetPhiFromFieldFunctions");

  std::vector<size_t> m_ids_to_copy = m_indices;
  std::vector<size_t> g_ids_to_copy = g_indices;
  if (m_indices.empty())
    for (size_t m = 0; m < num_moments_; ++m)
      m_ids_to_copy.push_back(m);
  if (g_ids_to_copy.empty())
    for (size_t g = 0; g < num_groups_; ++g)
      g_ids_to_copy.push_back(g);

  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;

  for (const size_t m : m_ids_to_copy)
  {
    for (const size_t g : g_ids_to_copy)
    {
      const size_t ff_index = phi_field_functions_local_map_.at({g, m});
      const auto& ff_ptr = field_functions_.at(ff_index);
      const auto& ff_data = ff_ptr->GetLocalFieldVector();

      for (const auto& cell : grid_ptr_->local_cells)
      {
        const auto& cell_mapping = sdm.GetCellMapping(cell);
        const size_t num_nodes = cell_mapping.NumNodes();

        for (size_t i = 0; i < num_nodes; ++i)
        {
          const int64_t imapA = sdm.MapDOFLocal(cell, i);
          const int64_t imapB = sdm.MapDOFLocal(cell, i, phi_uk_man, m, g);

          if (which_phi == PhiSTLOption::PHI_OLD)
            phi_old_local_[imapB] = ff_data[imapA];
          else if (which_phi == PhiSTLOption::PHI_NEW)
            phi_new_local_[imapB] = ff_data[imapA];
        } // for node
      }   // for cell
    }     // for g
  }       // for m
}

double
LBSSolver::ComputeFissionProduction(const std::vector<double>& phi)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::ComputeFissionProduction");

  const int first_grp = groups_.front().id;
  const int last_grp = groups_.back().id;

  // Loop over local cells
  double local_production = 0.0;
  for (auto& cell : grid_ptr_->local_cells)
  {
    const auto& transport_view = cell_transport_views_[cell.local_id];
    const auto& cell_matrices = unit_cell_matrices_[cell.local_id];

    // Obtain xs
    const auto& xs = transport_view.XS();
    const auto& F = xs.ProductionMatrix();
    const auto& nu_delayed_sigma_f = xs.NuDelayedSigmaF();

    if (not xs.IsFissionable())
      continue;

    // Loop over nodes
    const int num_nodes = transport_view.NumNodes();
    for (int i = 0; i < num_nodes; ++i)
    {
      const size_t uk_map = transport_view.MapDOF(i, 0, 0);
      const double IntV_ShapeI = cell_matrices.intV_shapeI(i);

      // Loop over groups
      for (size_t g = first_grp; g <= last_grp; ++g)
      {
        const auto& prod = F[g];
        for (size_t gp = 0; gp <= last_grp; ++gp)
          local_production += prod[gp] * phi[uk_map + gp] * IntV_ShapeI;

        if (options_.use_precursors)
          for (unsigned int j = 0; j < xs.NumPrecursors(); ++j)
            local_production += nu_delayed_sigma_f[g] * phi[uk_map + g] * IntV_ShapeI;
      }
    } // for node
  }   // for cell

  // Allreduce global production
  double global_production = 0.0;
  mpi_comm.all_reduce(local_production, global_production, mpi::op::sum<double>());

  return global_production;
}

double
LBSSolver::ComputeFissionRate(const std::vector<double>& phi)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::ComputeFissionRate");

  const int first_grp = groups_.front().id;
  const int last_grp = groups_.back().id;

  // Loop over local cells
  double local_fission_rate = 0.0;
  for (auto& cell : grid_ptr_->local_cells)
  {
    const auto& transport_view = cell_transport_views_[cell.local_id];
    const auto& cell_matrices = unit_cell_matrices_[cell.local_id];

    // Obtain xs
    const auto& xs = transport_view.XS();
    const auto& sigma_f = xs.SigmaFission();

    // skip non-fissionable material
    if (not xs.IsFissionable())
      continue;

    // Loop over nodes
    const int num_nodes = transport_view.NumNodes();
    for (int i = 0; i < num_nodes; ++i)
    {
      const size_t uk_map = transport_view.MapDOF(i, 0, 0);
      const double IntV_ShapeI = cell_matrices.intV_shapeI(i);

      // Loop over groups
      for (size_t g = first_grp; g <= last_grp; ++g)
        local_fission_rate += sigma_f[g] * phi[uk_map + g] * IntV_ShapeI;
    } // for node
  }   // for cell

  // Allreduce global production
  double global_fission_rate = 0.0;
  mpi_comm.all_reduce(local_fission_rate, global_fission_rate, mpi::op::sum<double>());

  return global_fission_rate;
}

void
LBSSolver::ComputePrecursors()
{
  CALI_CXX_MARK_SCOPE("LBSSolver::ComputePrecursors");

  const size_t J = max_precursors_per_material_;

  precursor_new_local_.assign(precursor_new_local_.size(), 0.0);

  // Loop over cells
  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& fe_values = unit_cell_matrices_[cell.local_id];
    const auto& transport_view = cell_transport_views_[cell.local_id];
    const double cell_volume = transport_view.Volume();

    // Obtain xs
    const auto& xs = transport_view.XS();
    const auto& precursors = xs.Precursors();
    const auto& nu_delayed_sigma_f = xs.NuDelayedSigmaF();

    // Loop over precursors
    for (uint64_t j = 0; j < xs.NumPrecursors(); ++j)
    {
      size_t dof = cell.local_id * J + j;
      const auto& precursor = precursors[j];
      const double coeff = precursor.fractional_yield / precursor.decay_constant;

      // Loop over nodes
      for (int i = 0; i < transport_view.NumNodes(); ++i)
      {
        const size_t uk_map = transport_view.MapDOF(i, 0, 0);
        const double node_V_fraction = fe_values.intV_shapeI(i) / cell_volume;

        // Loop over groups
        for (unsigned int g = 0; g < groups_.size(); ++g)
          precursor_new_local_[dof] +=
            coeff * nu_delayed_sigma_f[g] * phi_new_local_[uk_map + g] * node_V_fraction;
      } // for node i
    }   // for precursor j

  } // for cell
}

void
LBSSolver::SetPhiVectorScalarValues(std::vector<double>& phi_vector, double value)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetPhiVectorScalarValues");

  const size_t first_grp = groups_.front().id;
  const size_t final_grp = groups_.back().id;
  const auto& sdm = *discretization_;

  for (const auto& cell : grid_ptr_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t dof_map = sdm.MapDOFLocal(cell, i, flux_moments_uk_man_, 0, 0);

      double* phi = &phi_vector[dof_map];

      for (size_t g = first_grp; g <= final_grp; ++g)
        phi[g] = value;
    }
  }
}

void
LBSSolver::ScalePhiVector(PhiSTLOption which_phi, double value)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::ScalePhiVector");

  std::vector<double>* y_ptr;
  switch (which_phi)
  {
    case PhiSTLOption::PHI_NEW:
      y_ptr = &phi_new_local_;
      break;
    case PhiSTLOption::PHI_OLD:
      y_ptr = &phi_old_local_;
      break;
    default:
      throw std::logic_error("SetGSPETScVecFromPrimarySTLvector");
  }

  Scale(*y_ptr, value);
}

void
LBSSolver::SetGSPETScVecFromPrimarySTLvector(const LBSGroupset& groupset,
                                             Vec x,
                                             PhiSTLOption which_phi)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetGSPETScVecFromPrimarySTLvector");

  const std::vector<double>* y_ptr;
  switch (which_phi)
  {
    case PhiSTLOption::PHI_NEW:
      y_ptr = &phi_new_local_;
      break;
    case PhiSTLOption::PHI_OLD:
      y_ptr = &phi_old_local_;
      break;
    default:
      throw std::logic_error("SetGSPETScVecFromPrimarySTLvector");
  }

  double* x_ref;
  VecGetArray(x, &x_ref);

  int gsi = groupset.groups.front().id;
  int gsf = groupset.groups.back().id;
  int gss = gsf - gsi + 1;

  int64_t index = -1;
  for (const auto& cell : grid_ptr_->local_cells)
  {
    auto& transport_view = cell_transport_views_[cell.local_id];

    for (int i = 0; i < cell.vertex_ids.size(); ++i)
    {
      for (int m = 0; m < num_moments_; ++m)
      {
        size_t mapping = transport_view.MapDOF(i, m, gsi);
        for (int g = 0; g < gss; ++g)
        {
          index++;
          x_ref[index] = (*y_ptr)[mapping + g]; // Offset on purpose
        }                                       // for g
      }                                         // for moment
    }                                           // for dof
  }                                             // for cell

  VecRestoreArray(x, &x_ref);
}

void
LBSSolver::SetPrimarySTLvectorFromGSPETScVec(const LBSGroupset& groupset,
                                             Vec x,
                                             PhiSTLOption which_phi)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetPrimarySTLvectorFromGSPETScVec");

  std::vector<double>* y_ptr;
  switch (which_phi)
  {
    case PhiSTLOption::PHI_NEW:
      y_ptr = &phi_new_local_;
      break;
    case PhiSTLOption::PHI_OLD:
      y_ptr = &phi_old_local_;
      break;
    default:
      throw std::logic_error("SetPrimarySTLvectorFromGSPETScVec");
  }

  const double* x_ref;
  VecGetArrayRead(x, &x_ref);

  int gsi = groupset.groups.front().id;
  int gsf = groupset.groups.back().id;
  int gss = gsf - gsi + 1;

  int64_t index = -1;
  for (const auto& cell : grid_ptr_->local_cells)
  {
    auto& transport_view = cell_transport_views_[cell.local_id];

    for (int i = 0; i < cell.vertex_ids.size(); ++i)
    {
      for (int m = 0; m < num_moments_; ++m)
      {
        size_t mapping = transport_view.MapDOF(i, m, gsi);
        for (int g = 0; g < gss; ++g)
        {
          index++;
          (*y_ptr)[mapping + g] = x_ref[index];
        } // for g
      }   // for moment
    }     // for dof
  }       // for cell

  VecRestoreArrayRead(x, &x_ref);
}

void
LBSSolver::GSScopedCopyPrimarySTLvectors(const LBSGroupset& groupset,
                                         const std::vector<double>& x,
                                         std::vector<double>& y)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::GSScopedCopyPrimarySTLvectors");

  int gsi = groupset.groups.front().id;
  size_t gss = groupset.groups.size();

  for (const auto& cell : grid_ptr_->local_cells)
  {
    auto& transport_view = cell_transport_views_[cell.local_id];

    for (int i = 0; i < cell.vertex_ids.size(); ++i)
    {
      for (int m = 0; m < num_moments_; ++m)
      {
        size_t mapping = transport_view.MapDOF(i, m, gsi);
        for (int g = 0; g < gss; ++g)
        {
          y[mapping + g] = x[mapping + g];
        } // for g
      }   // for moment
    }     // for dof
  }       // for cell
}

void
LBSSolver::GSScopedCopyPrimarySTLvectors(const LBSGroupset& groupset,
                                         PhiSTLOption from_which_phi,
                                         PhiSTLOption to_which_phi)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::GSScopedCopyPrimarySTLvectors");

  std::vector<double>* y_ptr;
  switch (to_which_phi)
  {
    case PhiSTLOption::PHI_NEW:
      y_ptr = &phi_new_local_;
      break;
    case PhiSTLOption::PHI_OLD:
      y_ptr = &phi_old_local_;
      break;
    default:
      throw std::logic_error("GSScopedCopyPrimarySTLvectors");
  }

  std::vector<double>* x_src_ptr;
  switch (from_which_phi)
  {
    case PhiSTLOption::PHI_NEW:
      x_src_ptr = &phi_new_local_;
      break;
    case PhiSTLOption::PHI_OLD:
      x_src_ptr = &phi_old_local_;
      break;
    default:
      throw std::logic_error("GSScopedCopyPrimarySTLvectors");
  }

  int gsi = groupset.groups.front().id;
  size_t gss = groupset.groups.size();

  for (const auto& cell : grid_ptr_->local_cells)
  {
    auto& transport_view = cell_transport_views_[cell.local_id];

    for (int i = 0; i < cell.vertex_ids.size(); ++i)
    {
      for (int m = 0; m < num_moments_; ++m)
      {
        size_t mapping = transport_view.MapDOF(i, m, gsi);
        for (int g = 0; g < gss; ++g)
        {
          (*y_ptr)[mapping + g] = (*x_src_ptr)[mapping + g];
        } // for g
      }   // for moment
    }     // for dof
  }       // for cell
}

void
LBSSolver::SetGroupScopedPETScVecFromPrimarySTLvector(int first_group_id,
                                                      int last_group_id,
                                                      Vec x,
                                                      const std::vector<double>& y)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetGroupScopedPETScVecFromPrimarySTLvector");

  double* x_ref;
  VecGetArray(x, &x_ref);

  int gsi = first_group_id;
  int gsf = last_group_id;
  int gss = gsf - gsi + 1;

  int64_t index = -1;
  for (const auto& cell : grid_ptr_->local_cells)
  {
    auto& transport_view = cell_transport_views_[cell.local_id];

    for (int i = 0; i < cell.vertex_ids.size(); ++i)
    {
      for (int m = 0; m < num_moments_; ++m)
      {
        size_t mapping = transport_view.MapDOF(i, m, gsi);
        for (int g = 0; g < gss; ++g)
        {
          index++;
          x_ref[index] = y[mapping + g]; // Offset on purpose
        }                                // for g
      }                                  // for moment
    }                                    // for dof
  }                                      // for cell

  VecRestoreArray(x, &x_ref);
}

void
LBSSolver::SetPrimarySTLvectorFromGroupScopedPETScVec(int first_group_id,
                                                      int last_group_id,
                                                      Vec x,
                                                      std::vector<double>& y)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetPrimarySTLvectorFromGroupScopedPETScVec");

  const double* x_ref;
  VecGetArrayRead(x, &x_ref);

  int gsi = first_group_id;
  int gsf = last_group_id;
  int gss = gsf - gsi + 1;

  int64_t index = -1;
  for (const auto& cell : grid_ptr_->local_cells)
  {
    auto& transport_view = cell_transport_views_[cell.local_id];

    for (int i = 0; i < cell.vertex_ids.size(); ++i)
    {
      for (int m = 0; m < num_moments_; ++m)
      {
        size_t mapping = transport_view.MapDOF(i, m, gsi);
        for (int g = 0; g < gss; ++g)
        {
          index++;
          y[mapping + g] = x_ref[index];
        } // for g
      }   // for moment
    }     // for dof
  }       // for cell

  VecRestoreArrayRead(x, &x_ref);
}

void
LBSSolver::SetMultiGSPETScVecFromPrimarySTLvector(const std::vector<int>& groupset_ids,
                                                  Vec x,
                                                  PhiSTLOption which_phi)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetMultiGSPETScVecFromPrimarySTLvector");

  const std::vector<double>* y_ptr;
  switch (which_phi)
  {
    case PhiSTLOption::PHI_NEW:
      y_ptr = &phi_new_local_;
      break;
    case PhiSTLOption::PHI_OLD:
      y_ptr = &phi_old_local_;
      break;
    default:
      throw std::logic_error("SetMultiGSPETScVecFromPrimarySTLvector");
  }

  double* x_ref;
  VecGetArray(x, &x_ref);

  int64_t index = -1;
  for (int gs_id : groupset_ids)
  {
    const auto& groupset = groupsets_.at(gs_id);

    int gsi = groupset.groups.front().id;
    int gsf = groupset.groups.back().id;
    int gss = gsf - gsi + 1;

    for (const auto& cell : grid_ptr_->local_cells)
    {
      auto& transport_view = cell_transport_views_[cell.local_id];

      for (int i = 0; i < cell.vertex_ids.size(); ++i)
      {
        for (int m = 0; m < num_moments_; ++m)
        {
          size_t mapping = transport_view.MapDOF(i, m, gsi);
          for (int g = 0; g < gss; ++g)
          {
            index++;
            x_ref[index] = (*y_ptr)[mapping + g]; // Offset on purpose
          }                                       // for g
        }                                         // for moment
      }                                           // for dof
    }                                             // for cell
  }                                               // for groupset id

  VecRestoreArray(x, &x_ref);
}

void
LBSSolver::SetPrimarySTLvectorFromMultiGSPETScVecFrom(const std::vector<int>& groupset_ids,
                                                      Vec x,
                                                      PhiSTLOption which_phi)
{
  CALI_CXX_MARK_SCOPE("LBSSolver::SetPrimarySTLvectorFromMultiGSPETScVecFrom");

  std::vector<double>* y_ptr;
  switch (which_phi)
  {
    case PhiSTLOption::PHI_NEW:
      y_ptr = &phi_new_local_;
      break;
    case PhiSTLOption::PHI_OLD:
      y_ptr = &phi_old_local_;
      break;
    default:
      throw std::logic_error("SetPrimarySTLvectorFromMultiGSPETScVecFrom");
  }

  const double* x_ref;
  VecGetArrayRead(x, &x_ref);

  int64_t index = -1;
  for (int gs_id : groupset_ids)
  {
    const auto& groupset = groupsets_.at(gs_id);

    int gsi = groupset.groups.front().id;
    int gsf = groupset.groups.back().id;
    int gss = gsf - gsi + 1;

    for (const auto& cell : grid_ptr_->local_cells)
    {
      auto& transport_view = cell_transport_views_[cell.local_id];

      for (int i = 0; i < cell.vertex_ids.size(); ++i)
      {
        for (int m = 0; m < num_moments_; ++m)
        {
          size_t mapping = transport_view.MapDOF(i, m, gsi);
          for (int g = 0; g < gss; ++g)
          {
            index++;
            (*y_ptr)[mapping + g] = x_ref[index];
          } // for g
        }   // for moment
      }     // for dof
    }       // for cell
  }         // for groupset id

  VecRestoreArrayRead(x, &x_ref);
}

} // namespace opensn
