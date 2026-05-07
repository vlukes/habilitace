"""
Two-scale nonlinear simulation of fluid-saturated hyperelastic solids.

Running simulation: sfepy-run fe2_perf_makro_dns.py
"""
import os.path as osp
import fe2_perf_makro
from csa_makro import merge_locals
from dns_hyper_perf import move_fun

wdir = osp.dirname(__file__)

time_stepping_fun = fe2_perf_makro.time_stepping_fun

def define(filename_mesh='meshes/macro_16x8.vtk',
           eps0=0.3/32, recovery_idxs=[(18, 0), (72, 0), (98, 0)],
           filename_mesh_micro='meshes/micro_perf_1ch.vtk',
           n_channels=1,
           t_end=0.1, t_nstep=50,
           displ_val=[0.03, 0],
           output_dir='output',
           multi=True,
           ):
    
    d = fe2_perf_makro.define(filename_mesh=filename_mesh, eps0=eps0,
                              filename_mesh_micro=filename_mesh_micro,
                              output_dir=output_dir,multi=multi,
                              n_channels=n_channels,
                              t_end=t_end, t_nstep=t_nstep,)


    functions = {
        'move': (move_fun,),
    }

    materials = {
        'load' : ({'val': [[0], [0]]},),
    }

    ebcs = {
        'fix_left': ('Left', {'u.all': 0.0}),
        'move_right': ('Right', {'u.all': 'move'}),
    }   

    idt = d['idt']
    del d['equations']['mass_conservation_2']
    
    equations = {
        'balance_of_forces': """
              dw_nonsym_elastic.i.Omega(solid.A, v, u)
            - dw_biot.i.Omega(solid.B1, v, p1)
            =
              dw_surface_ltr.i.Right(load.val, v)
            - dw_lin_prestress.i.Omega(solid.S, v)
            - dw_lin_prestress.i.Omega(solid.Q, v)
            """,
        'mass_conservation_1': f"""
    + {idt} * dw_biot.i.Omega(solid.B1, u, q1)
            + dw_diffusion.i.Omega(solid.C1, q1, p1)
            =
            - dw_volume_lvf.i.Omega(solid.Z1, q1)
            - dw_diffusion.i.Omega(solid.C1, q1, P1)
            """,
    }

    return merge_locals(locals(), d)
