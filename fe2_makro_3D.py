"""
Two-scale nonlinear simulation - FE2 approach.

Running simulation: sfepy-run fe2_makro_3D.py
"""
import os.path as osp
import numpy as nm
import fe2_makro as nonlin_macro
from csa_makro import merge_locals

wdir = osp.dirname(__file__)


def displ_fun(ts, coor, problem=None, **kwargs):
    from sfepy.linalg import rotation_matrix2d

    centre = nm.array([0, 0], dtype=nm.float64)
    vec = coor[:,1:3] - centre

    angle = problem.conf.angle * ts.nt
    print('angle:', angle)

    mtx = rotation_matrix2d(angle)
    vec_rotated = nm.dot(vec, mtx)

    displ = vec_rotated - vec

    return displ


def define(filename_mesh='meshes/macro_3D.vtk',
           filename_mesh_micro='meshes/micro_3D.vtk',
           output_dir='output_3D',
           angle=60.,
           eps_a=1e3,
           polar_decomposition=False,
           **kwargs):

    d = nonlin_macro.define(filename_mesh=filename_mesh,
                            filename_mesh_micro=filename_mesh_micro,
                            output_dir=output_dir, eps_a=eps_a,
                            polar_decomposition=polar_decomposition,
                            **kwargs)

    functions = {
        'displ_fun': (displ_fun,),
    }

    materials = {}

    ebcs = {
        'l': ('Left', {'u.all': 0.0}),
        'r': ('Right', {'u.0' : 0.0, 'u.[1,2]' : 'displ_fun'}),
    }

    equations = {
        'balance':
            'dw_ul_he_by_fun.i.Omega(get_homog, v, u) = 0'
    }

    return merge_locals(locals(), d)