#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard Reed 1D 1-group problem

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLProductQuadrature1DSlab
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSolver
    from pyopensn.logvol import RPPLogicalVolume

if __name__ == "__main__":

    try:
        print("Scattering Parameter = {}".format(scatt))
        param = scatt
    except:
        scatt=0.0
        print("Scattering Nominal = {}".format(scatt))

    try:
        print("Cross Section Parameter = {}".format(sigma_t))
        param = sigma_t
    except:
        sigma_t=1.0
        print("Cross Section Nominal = {}".format(sigma_t))

    try:
        print("Source Parameter = {}".format(param_q))
        param = param_q
    except:
        param_q=1.0
        print("Source Nominal = {}".format(param_q))

    try:
        print("Parameter id = {}".format(p_id))
    except:
        p_id=0
        print("Parameter id = {}".format(p_id))

    try:
        if phase == 0:
            print("Offline Phase")
            phase = "offline"
        elif phase == 1:
            print("Merge Phase")
            phase = "merge"
        elif phase == 2:
            print("Systems Phase")
            phase = "systems"
        elif phase == 3:
            print("Online Phase")
            phase = "online"
    except:
        phase="offline"
        print("Phase default to offline")
    
    # Create Mesh
    widths = [2., 1., 2., 1., 2.]
    nrefs = [200, 200, 200, 200, 200]
    Nmat = len(widths)
    nodes = [0.]
    for imat in range(Nmat):
        dx = widths[imat] / nrefs[imat]
        for i in range(nrefs[imat]):
            nodes.append(nodes[-1] + dx)
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes])
    grid = meshgen.Execute()

    # Set block IDs
    z_min = 0.0
    z_max = widths[1]
    for imat in range(Nmat):
        z_max = z_min + widths[imat]
        print("imat=", imat, ", zmin=", z_min, ", zmax=", z_max)
        lv = RPPLogicalVolume(infx=True, infy=True, zmin=z_min, zmax=z_max)
        grid.SetBlockIDFromLogicalVolume(lv, imat, True)
        z_min = z_max

    # Add cross sections to materials
    total = [50., 5., 0., sigma_t, sigma_t]
    c = [0., 0., 0., scatt, scatt]
    xs_map = len(total) * [None]
    for imat in range(Nmat):
        xs_ = MultiGroupXS()
        xs_.CreateSimpleOneGroup(total[imat], c[imat])
        xs_map[imat] = {
            "block_ids": [imat], "xs": xs_,
        }

    # Create sources in 1st and 4th materials
    src0 = VolumetricSource(block_ids=[0], group_strength=[50.])
    src1 = VolumetricSource(block_ids=[3], group_strength=[param_q])

    # Angular Quadrature
    gl_quad = GLProductQuadrature1DSlab(n_polar=128, scattering_order=0)

    # LBS block option
    num_groups = 1
    if phase == "online":
        phys_options = {
                "param_id": 0,
                "phase": phase,
                "param_file": "data/params.txt",
                "new_point": [scatt, param_q]
            }
    else:
        phys_options = {
                "param_id": p_id,
                "phase": phase
            }
        
    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": gl_quad,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-9,
                "l_max_its": 300,
                "gmres_restart_interval": 30,
            },
        ],
        xs_map=xs_map,
        scattering_order=0,
        options=phys_options,
        volumetric_sources= [src0, src1],
        boundary_conditions= [
                    {"name": "zmin", "type": "vacuum"},
                    {"name": "zmax", "type": "vacuum"}
        ]         
    )

    # Initialize and execute solver
    ss_solver = SteadyStateSourceSolver(problem=phys)
    ss_solver.Initialize()
    ss_solver.Execute()

    # compute particle balance
    phys.ComputeBalance()
    
    if phase == "online":
        phys.WriteFluxMoments("output/rom")
    if phase == "offline":
        phys.WriteFluxMoments("output/fom")