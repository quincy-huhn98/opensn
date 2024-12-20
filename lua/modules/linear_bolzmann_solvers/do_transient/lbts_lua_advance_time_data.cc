// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_transient_solver/lbts_transient_solver.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"

#if 0
#define PropertyArgCntErr(prop_name)                                                               \
  throw std::logic_error(fname + ": Insufficient amount of arguments, " +                          \
                         std::to_string(num_args) + ", for property " + prop_name)

namespace lbts_lua_utils
{

//###################################################################
/**Advances time dependent data to the next timestep.

\param SolverIndex int Handle to the solver..

*/
int
LBTSAdvanceTimeData(lua_State* L)
{
  const std::string fname = "LBTSSetProperty";
  const int num_args = lua_gettop(L);
  if (num_args != 1) LuaPostArgAmountError(fname, 1, num_args);

  // Get the solver
  LuaCheckNilValue(fname, L, 1);
  const int solver_handle = lua_tointeger(L, 1);

  auto& solver =
    opensn::GetStackItem<DiscOrdTransientSolver>(opensn::object_stack, solver_handle, fname);

  solver.Advance();

  return 0;
}

} // namespace lbts_lua_utils
#endif
