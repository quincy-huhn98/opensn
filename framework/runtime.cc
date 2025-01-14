// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/runtime.h"
#include "framework/event_system/system_wide_event_publisher.h"
#include "framework/post_processors/post_processor.h"
#include "framework/event_system/event.h"
#include "framework/math/math.h"
#include "framework/object_factory.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "config.h"
#include "caliper/cali.h"
#include "hdf5.h"
#include <iostream>
#ifdef OPENSN_WITH_LIBROM
#include "librom.h"
#endif

namespace opensn
{

// Global variables
Logger& log = Logger::GetInstance();
mpi::Communicator mpi_comm;
bool use_caliper = false;
std::string cali_config("runtime-report(calc.inclusive=true),max_column_width=80");
cali::ConfigManager cali_mgr;
Timer program_timer;
int current_mesh_handler = -1;
bool suppress_color = false;
std::filesystem::path input_path;

std::vector<std::shared_ptr<MeshContinuum>> mesh_stack;
std::vector<std::shared_ptr<SurfaceMesh>> surface_mesh_stack;
std::vector<std::shared_ptr<FieldFunctionInterpolation>> field_func_interpolation_stack;
std::vector<std::shared_ptr<UnpartitionedMesh>> unpartitionedmesh_stack;
std::vector<std::shared_ptr<Material>> material_stack;
std::vector<std::shared_ptr<MultiGroupXS>> multigroup_xs_stack;
std::vector<std::shared_ptr<FieldFunction>> field_function_stack;
std::vector<std::shared_ptr<AngularQuadrature>> angular_quadrature_stack;
std::vector<std::shared_ptr<Object>> object_stack;
std::vector<std::shared_ptr<SpatialDiscretization>> sdm_stack;
std::vector<std::shared_ptr<PostProcessor>> postprocessor_stack;
std::vector<std::shared_ptr<Function>> function_stack;

int
Initialize()
{
  if (use_caliper)
  {
    cali_mgr.add(cali_config.c_str());
    cali_set_global_string_byname("opensn.version", GetVersionStr().c_str());
    cali_set_global_string_byname("opensn.input", input_path.c_str());
    cali_mgr.start();
  }

  CALI_MARK_BEGIN(opensn::program.c_str());

  SystemWideEventPublisher::GetInstance().PublishEvent(Event("ProgramStart"));

  // Disable internal HDF error reporting
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

// #ifdef OPENSN_WITH_LIBROM
//     // small placeholder example to verify libROM is working
//     int rank = 0;
//     int dim = 6;

//     // Construct the incremental basis generator to use the fast update
//     // incremental algorithm and the incremental sampler.
//     CAROM::Options svd_options = CAROM::Options(dim, 2).setMaxBasisDimension(2)
//                                  .setIncrementalSVD(1.0e-2, 1.0e-6, 1.0e-2, 0.11, true).setDebugMode(true);


//     CAROM::BasisGenerator inc_basis_generator(
//         svd_options, true
//     );

//     // Define the values for the first sample.
//     double vals0[6] = {1.0, 6.0, 3.0, 8.0, 17.0, 9.0};

//     // Define the values for the second sample.
//     double vals1[6] = {2.0, 7.0, 4.0, 9.0, 18.0, 10.0};

//     bool status = false;

//     // Take the first sample.
//     if (inc_basis_generator.isNextSample(0.0)) {
//         status = inc_basis_generator.takeSample(&vals0[dim*rank]);
//         if (status) {
//             inc_basis_generator.computeNextSampleTime(&vals0[dim*rank],
//                     &vals0[dim*rank],
//                     0.0);
//         }
//     }

//     // Take the second sample.
//     if (status && inc_basis_generator.isNextSample(0.11)) {
//         status = inc_basis_generator.takeSample(&vals1[dim*rank]);
//         if (status) {
//             inc_basis_generator.computeNextSampleTime(&vals1[dim*rank],
//                     &vals1[dim*rank],
//                     0.11);
//         }
//     }
// #endif

  return 0;
}

void
Finalize()
{
  SystemWideEventPublisher::GetInstance().PublishEvent(Event("ProgramExecuted"));
  mesh_stack.clear();
  surface_mesh_stack.clear();
  field_func_interpolation_stack.clear();
  unpartitionedmesh_stack.clear();
  material_stack.clear();
  multigroup_xs_stack.clear();
  function_stack.clear();
  object_stack.clear();

  CALI_MARK_END(opensn::program.c_str());
}

void
Exit(int error_code)
{
  mpi_comm.abort(error_code);
}

std::string
GetVersionStr()
{
  return PROJECT_VERSION;
}

} // namespace opensn
