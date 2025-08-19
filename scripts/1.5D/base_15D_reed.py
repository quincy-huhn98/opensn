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
            print("MI-POD")
            phase = "mipod"
        elif phase == 4:
            print("Online Phase")
            phase = "online"
    except:
        phase="offline"
        print("Phase default to offline")
    
    # Setup mesh
    widths = [2., 1., 2., 1., 2.]
    nrefs = [40, 20, 40, 20, 40]
    Nmat = len(widths)
    x_nodes = [0.]
    for imat in range(Nmat):
        dx = widths[imat] / nrefs[imat]
        for i in range(nrefs[imat]):
            x_nodes.append(x_nodes[-1] + dx)
    
    y_nodes = []
    N = 80
    L = 8
    xmin = 0
    dx = L / N
    for i in range(N + 1):
        y_nodes.append(xmin + i * dx)
    meshgen = OrthogonalMeshGenerator(
        node_sets=[x_nodes, y_nodes],
        partitioner=KBAGraphPartitioner(
            nx=2,
            ny=2,
            xcuts=[4.0],
            ycuts=[4.0],
        ))
    grid = meshgen.Execute()

    x_min = 0.0
    x_max = widths[1]
    for imat in range(Nmat):
        x_max = x_min + widths[imat]
        lv = RPPLogicalVolume(xmin=x_min, xmax=x_max, infy=True, infz=True)
        grid.SetBlockIDFromLogicalVolume(lv, imat, True)
        x_min = x_max


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

    # Setup Physics
    fac = 1
    #pquad = GLCProductQuadrature2DXY(6 * fac, 16 * fac)
    pquad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=32, scattering_order=0)

    # LBS block option
    if phase == "online":
        phys_options = {
                "boundary_conditions": [
                    {"name": "zmin", "type": "vacuum"},
                    {"name": "zmax", "type": "vacuum"},
                    {"name": "ymin", "type": "vacuum"},
                    {"name": "ymax", "type": "vacuum"}
                ],
                "volumetric_sources": [src0, src1],
                "param_id": 0,
                "phase": phase,
                "param_file": "data/sigmas.txt",
                "new_point": param
            }
    else:
        phys_options = {
                "boundary_conditions": [
                    {"name": "zmin", "type": "vacuum"},
                    {"name": "zmax", "type": "vacuum"},
                    {"name": "ymin", "type": "vacuum"},
                    {"name": "ymax", "type": "vacuum"}
                ],
                "volumetric_sources": [src0, src1],
                "param_id": p_id,
                "phase": phase
            }
    
    num_groups = 1
    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": [0, 0],
                "angular_quadrature": pquad,
                "angle_aggregation_num_subsets": 1,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-8,
                "l_max_its": 300,
                "gmres_restart_interval": 100,
            },
        ],
        xs_map=xs_map,
        scattering_order= 0,
        options=phys_options
    )

    # Initialize and execute solver
    ss_solver = SteadyStateSolver(lbs_problem=phys)
    ss_solver.Initialize()
    ss_solver.Execute()

    # compute particle balance
    phys.ComputeBalance()
    
    if phase == "online":
        phys.WriteFluxMoments("output/rom")
    if phase == "mipod":
        phys.WriteFluxMoments("output/mi_rom")
    if phase == "offline":
        phys.WriteFluxMoments("output/fom")