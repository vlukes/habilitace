"""
Two-scale nonlinear simulation - FE2 approach.
"""
import os.path as osp
import fe2_makro
from csa_makro import merge_locals
from dns_hyper import move_fun

wdir = osp.dirname(__file__)


def define(filename_mesh='meshes/macro_16x8.vtk',
           filename_mesh_micro='meshes/micro_2.vtk',
           n_step=80,
           eps_a=1e3,
           eps_r=1e-3,
           displ_val=[0.03, 0],
           recovery_idxs=[(18, 0), (72, 0), (98, 0)],
           **kwargs):

    d = fe2_makro.define(filename_mesh=filename_mesh,
                         filename_mesh_micro=filename_mesh_micro,
                         n_step=n_step,eps_a=eps_a,eps_r=eps_r,
                         recovery_idxs=recovery_idxs,
                         polar_decomposition=False,save_qp=True,
                         integration='full',
                         **kwargs)

    functions = {
        'move': (move_fun,),
    }

    ebcs = {
        'fix_left': ('Left', {'u.all': 0.0}),
        'move_right': ('Right', {'u.all': 'move'}),
    }

    equations = {
        'balance': 'dw_ul_he_by_fun.i.Omega(get_homog, v, u) = 0'
    }

    return merge_locals(locals(), d)