"""
Two-scale nonlinear simulation - microscopic subproblem.
"""
import os.path as osp
from functools import partial
import numpy as nm
from sfepy.homogenization.utils import define_box_regions
import sfepy.homogenization.coefs_base as cb
import sfepy.discrete.fem.periodic as per
from sfepy.base.base import Struct
from sfepy.terms.terms_hyperelastic_ul import\
    HyperElasticULFamilyData, NeoHookeanULTerm, BulkPenaltyULTerm
from sfepy.terms.extmods.terms import sym2nonsym
from sfepy.discrete.functions import ConstantFunctionByRegion
import sfepy.linalg as la
from sfepy.base import multiproc
from sfepy.base.base import output, get_default

wdir = osp.dirname(__file__)
mp_module, _ = multiproc.get_multiproc()
multiproc_dependecies = mp_module.get_dict('dependecies', clear=True)


class CorrEquilibrium(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        if self.eps_a is not None:
            problem.set_equations(self.equations)
            problem.select_bcs(ebc_names=self.ebcs, epbc_names=self.epbcs,
                            lcbc_names=self.get('lcbcs', []))
            problem.init_solvers()

            coors0 = problem.get_mesh_coors(actual=True).copy()
            coors = coors0.copy()
            
            iiter = 1
            residuum = 1.
            while residuum > self.eps_a:
                state = problem.solve(save_results=False)
                du = state()
                residuum = nm.linalg.norm(du)

                coors += du.reshape(coors.shape)
                problem.set_mesh_coors(coors, actual=True)

                output(f'micro equilibrium - iter: {iiter}, residuum: {residuum}')
                iiter += 1

            du = problem.get_mesh_coors(actual=True) - coors0

        else:
            iiter = 0
            du = problem.get_mesh_coors() * 0

        return cb.CorrSolution(name='update_coors', state=du, niter=iiter)


def save_micro_state(pb, file_tag, displ, strain, stress, mtx_r):
    oStruct = partial(Struct, name='output_data', dofs=None)
    out = {}

    if mtx_r is None:
        flag = ''
    else:
        flag = '_no_rot'
        coors0 = pb.domain.get_mesh_coors()
        coors1 = coors0.copy()
        coors1 += displ
        coors1 = nm.dot(coors1, mtx_r.T)
        displ = coors1 - coors0

    out['cauchy_stress' + flag] = oStruct(mode='cell', data=stress)
    out['green_strain' + flag] = oStruct(mode='cell', data=strain)
    out['displacement'] = oStruct(mode='vertex', data=displ)

    output_dir = pb.conf.options.get('output_dir', '.')
    micro_name = pb.get_output_name(extra=f'recovered_{file_tag}')
    filename = osp.join(output_dir, osp.basename(micro_name))
    pb.save_state(filename, out=out)


def get_hyperelastic_mat(ts, coors, term, pb, dv=None):
    if not hasattr(pb, 'family_data'):
        pb.family_data = HyperElasticULFamilyData()

    state_u = pb.create_variables(['U1'])['U1']

    du = pb.domain.get_mesh_coors(actual=True) - pb.domain.get_mesh_coors()

    if dv is not None:
        du += dv
    
    state_u.set_data(du)
    state_u.field.clear_mappings()
    fd = pb.family_data(state_u, term.region, term.integral,
                        list(term.geometry_types.values())[0])

    if len(state_u.field.mappings0) == 0:
        state_u.field.save_mappings()

    n_el, n_qp, dim, _, _ = state_u.get_data_shape(term.integral,
                                                   term.act_integration,
                                                   term.region.name)

    if not hasattr(pb, 'material_data'):
        conf_mat = pb.conf.materials
        solid_key = [key for key in conf_mat.keys() if 'solid' in key][0]
        solid_mat = conf_mat[solid_key].values
        mat = {}
        for mat_key in ['mu', 'K']:
            mat_fun = ConstantFunctionByRegion({mat_key: solid_mat[mat_key]})
            mat[mat_key] = mat_fun.function(ts=ts, coors=coors, mode='qp',
                term=term, problem=pb)[mat_key].reshape((n_el, n_qp, 1, 1))
        
        pb.material_data = mat

    mat = pb.material_data

    shape = fd.green_strain.shape[:2]
    sym = fd.green_strain.shape[-2]
    dim2 = dim**2

    fargs = [fd.get(name) for name in NeoHookeanULTerm.family_data_names]
    stress = nm.empty(shape + (sym, 1), dtype=nm.float64)
    tanmod = nm.empty(shape + (sym, sym), dtype=nm.float64)
    NeoHookeanULTerm.stress_function(stress, mat['mu'], *fargs)
    NeoHookeanULTerm.tan_mod_function(tanmod, mat['mu'], *fargs)

    fargs = [fd.get(name) for name in BulkPenaltyULTerm.family_data_names]
    stress_p = nm.empty(shape + (sym, 1), dtype=nm.float64)
    tanmod_p = nm.empty(shape + (sym, sym), dtype=nm.float64)
    BulkPenaltyULTerm.stress_function(stress_p, mat['K'], *fargs)
    BulkPenaltyULTerm.tan_mod_function(tanmod_p, mat['K'], *fargs)

    stress_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    tanmod_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    sym2nonsym(stress_ns, stress + stress_p)
    sym2nonsym(tanmod_ns, tanmod + tanmod_p)

    mtx_j = fd.det_f
    tanmod_ns = ((tanmod_ns + stress_ns) / mtx_j).reshape((-1, dim2, dim2))
    stress = ((stress + stress_p) / mtx_j).reshape((-1, sym, 1))

    mtx_f = fd.mtx_f.reshape((-1, dim, dim))

    micro_state, im = pb.micro_state

    if 'id' in micro_state:
        mac_id = micro_state['id'][im]

        if mac_id in pb.conf.recovery_idxs:
            macro_data = pb.homogenization_macro_data
            mac_step, mac_iter = macro_data['step_iter']
            astress = nm.average(stress.reshape((n_el, n_qp, sym, 1)),
                                axis=1)[:, None, :, :]
            astrain = nm.average(fd.green_strain, axis=1)[:, None, :, :]
            flag = f'{mac_id:03d}.{(mac_step * 100 + mac_iter):05d}'
            if macro_data['mtx_r'] is not None:
                mtx_r = macro_data['mtx_r'][im]
            else:
                mtx_r = None
        
            save_micro_state(pb, flag, du, astrain, astress, mtx_r)

    return tanmod_ns, stress, mtx_f


def def_mat(ts, coors, mode=None, term=None, problem=None, **kwargs):
    if mode != 'qp':
        return None

    tanmod, stress, mtx_f = get_hyperelastic_mat(ts, coors, term, problem)

    dim = mtx_f.shape[-1]
    out = {
        'E': 0.5 * (la.dot_sequences(mtx_f, mtx_f, 'ATB') - nm.eye(dim)),
        'A': tanmod,
        'S': stress,
    }

    return out


def define(dim, filename_mesh, output_dir='output', multiprocessing=True,
           recovery_idxs=[], equilibrium_eps=None):
    filename_mesh = osp.join(wdir, filename_mesh)

    options = {
        'coefs': 'coefs',
        'requirements': 'requirements',
        'volume': {'expression': 'ev_volume.5.Y(u)'},
        'output_dir': output_dir,
        'coefs_filename': 'coefs_hyper_homog',
        'multiprocessing': multiprocessing,
        'micro_update': {'coors': [('corrs_rs', 'u', 'mtx_e')]},
        'mesh_update_variable': 'u',
        'recovery_hook': 'recovery_hook',
    }

    fields = {
        'displacement': ('real', 'vector', 'Y', 1),
    }

    integrals = {
        'i': 2,
    }

    functions = {
        'match_x_plane': (per.match_x_plane,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
        'mat_fce': (def_mat,),
    }

    materials = {
        'mat_he': 'mat_fce',
        # E, nu, G, K (GPa)
        # glass:    70.     0.23    28.46   43.21
        # epoxy:    3.76    0.39    1.35    5.70
        'solid': ({'K': {'Ym': 5.7e9, 'Yc': 43.21e9},
                   'mu': {'Ym': 1.35e9, 'Yc': 28.46e9},
                  },),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'Pi': ('parameter field', 'displacement', 'u'),
        'U1': ('parameter field', 'displacement', '(set-to-None)'),
        'U2': ('parameter field', 'displacement', '(set-to-None)'),
    }

    regions = {
        'Y': 'all',
        'Ym': 'cells of group 1',
        'Yc': 'cells of group 2',
    }

    regions.update(define_box_regions(dim, (0, 0, 0)[:dim], (1, 1, 1)[:dim]))

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
    }

    epbcs = {
        'periodic_ux': (['Left', 'Right'], {'u.all': 'u.all'}, 'match_x_plane'),
    }

    periodic_all = ['periodic_ux', 'periodic_uy']

    if dim == 3:
        epbcs.update({
            'periodic_uy': (['Near', 'Far'], {'u.all': 'u.all'}, 'match_y_plane'),
            'periodic_uz': (['Bottom', 'Top'], {'u.all': 'u.all'}, 'match_z_plane'),
        })
        periodic_all += ['periodic_uz']
    else:
        epbcs.update({
            'periodic_uy': (['Bottom', 'Top'], {'u.all': 'u.all'}, 'match_y_plane'),
        })

    coefs = {
        'A': {
            'requires': ['pis_u', 'corrs_rs'],
            'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, U1, U2)',
            'set_variables': [('U1', ('pis_u', 'corrs_rs'), 'u'),
                              ('U2', ('pis_u', 'corrs_rs'), 'u')],
            'class': cb.CoefNonSymNonSym,
        },
        'S': {
            'requires': ['c.A'],
            'expression': 'ev_integrate_mat.i.Y(mat_he.S, u)',
            'class': cb.CoefOne,
        },
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'corrs_rs': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u'],
            'epbcs': periodic_all,
            'equations': {
                'balance':
                    """dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                   = - dw_nonsym_elastic.i.Y(mat_he.A, v, Pi)"""
            },
            'set_variables': [('Pi', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_hyper_homog',
        },
        'equilibrium': {
            'ebcs': ['fixed_u'],
            'epbcs': periodic_all,
            'equations': {
                'balance_equilibrium':
                    """dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                    = - dw_lin_prestress.i.Y(mat_he.S, v)"""
            },
            'class': CorrEquilibrium,
            'eps_a': equilibrium_eps,
        }
    }

    if equilibrium_eps is not None:
        requirements['corrs_rs']['requires'].append('equilibrium')

    solvers = {
        'ls': ('ls.mumps', {'use_presolve': True}),
        'nls': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-4,
            'problem': 'nonlinear',
        }),
    }

    return locals()
