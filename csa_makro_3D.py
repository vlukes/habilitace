"""
Two-scale nonlinear simulation - FE2 approach.
"""
import csa_makro as nonlin_macro
from fe2_makro_3D import displ_fun

def define(delta=0.01,
           err_idxs=[],
           n_maxiter=20,
           filename_mesh='meshes/macro_3D.vtk',
           filename_mesh_micro='meshes/micro_3D.vtk',
           output_dir='output_3D',
           angle=35.,
           eps_a=1e3,
           **kwargs):

    d = nonlin_macro.define(filename_mesh=filename_mesh,
                            filename_mesh_micro=filename_mesh_micro,
                            output_dir=output_dir,
                            eps_a=eps_a,
                            n_maxiter=n_maxiter, err_idxs=err_idxs,
                            delta=delta,
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
            'dw_ul_he_by_fun.i.Omega(get_homog_clusters, v, u) = 0'
    }

    return nonlin_macro.merge_locals(locals(), d)