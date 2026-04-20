"""
Nonlinear simulation of hyperelastic structure undergoing finite deformations.

Running simulation: sfepy-run dns_hyper.py
"""
import os.path as osp
from functools import partial
import numpy as nm
import meshio
from sfepy.base.base import Struct, output
from sfepy.terms.terms_hyperelastic_ul import (NeoHookeanULTerm,
    BulkPenaltyULTerm)
from sfepy.terms.extmods.terms import sym2nonsym
from fe2_makro import (ulf_init, ulf_iteration_hook, hyperelastic_data)

wdir = osp.dirname(__file__)


def get_mat_from_cache(n_qp):
    macro = hyperelastic_data['macro']
    pb = macro['problem']

    field = pb.fields['displacement']
    if len(field.mappings0) == 0:
        field.save_mappings()

    if 'mat_mu_K' not in macro:   
        conf_mat = pb.conf.materials
        solid_key = [key for key in conf_mat.keys() if 'solid' in key][0]
        solid_mat = conf_mat[solid_key].values
        mat = {}
        n_el = len(pb.domain.regions['Omega'].entities[-1])
        for mat_key, mat_val in solid_mat.items():
            mat[mat_key] = nm.zeros((n_el, n_qp, 1, 1), dtype=nm.float64)
            for rn, rv in mat_val.items():
                rcells = pb.domain.regions[rn].entities[-1]
                mat[mat_key][rcells, :, 0, 0] = rv

        macro['mat_mu_K'] = mat

    return macro['mat_mu_K']


def get_hyperelastic_mat(family_data, mode):
    n_el, n_qp, sym, _= family_data.green_strain.shape
    dim = family_data.mtx_f.shape[-1]
    dim2 = dim**2

    mat = get_mat_from_cache(n_qp)

    shape = family_data.green_strain.shape[:2]
    sym = family_data.green_strain.shape[-2]

    fargs = [family_data.get(name)
             for name in NeoHookeanULTerm.family_data_names]
    stress = nm.empty(shape + (sym, 1), dtype=nm.float64)
    tanmod = nm.empty(shape + (sym, sym), dtype=nm.float64)
    NeoHookeanULTerm.stress_function(stress, mat['mu'], *fargs)
    NeoHookeanULTerm.tan_mod_function(tanmod, mat['mu'], *fargs)

    fargs = [family_data.get(name)
             for name in BulkPenaltyULTerm.family_data_names]
    stress_p = nm.empty(shape + (sym, 1), dtype=nm.float64)
    tanmod_p = nm.empty(shape + (sym, sym), dtype=nm.float64)
    BulkPenaltyULTerm.stress_function(stress_p, mat['K'], *fargs)
    BulkPenaltyULTerm.tan_mod_function(tanmod_p, mat['K'], *fargs)

    stress_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    tanmod_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    sym2nonsym(stress_ns, stress + stress_p)
    sym2nonsym(tanmod_ns, tanmod + tanmod_p)

    if mode == 'tan_mod':
        out = tanmod_ns + stress_ns
    elif mode == 'stress':
        out = stress + stress_p
    else:
        raise ValueError()

    return out


def post_process(out, pb, state, extend=False):
    ev = partial(pb.evaluate, mode='el_avg', get_homog=pb.conf.get_homog)
    stress = ev('dw_ul_he_by_fun.i.Omega(get_homog, v, u)', term_mode='stress')
    strain = ev('dw_ul_he_by_fun.i.Omega(get_homog, v, u)', term_mode='strain')
    out['cauchy_stress'] = Struct(name='output_data', mode='cell', data=stress)
    out['green_strain'] = Struct(name='output_data', mode='cell', data=strain)

    hyperelastic_data['macro']['niter'].append(pb.iiter)

    vol = ev('ev_volume.i.Omega(u)')
    eids = pb.conf.mmesh.cell_data['eid'][0]
    astress = stress * vol
    for eid in nm.unique(eids):
        idxs = eids == eid
        astress[idxs] = astress[idxs].sum(axis=0) / vol[idxs].sum(axis=0)

    out['acauchy_stress'] = Struct(name='output_data', mode='cell', data=astress)
    out['eid'] = Struct(name='output_data', mode='cell', data=eids[:, None, None, None])

    return out


def post_process_final(pb, state, out={}):
    macro_data = hyperelastic_data['macro']
    output(f'>>> num. iterations: {macro_data["niter"]}')


def move_fun(ts, coor, **kwargs):
    pb = kwargs['problem']
    displ = nm.array(pb.conf.displ_val) * ts.nt

    return coor * 0 + displ


def define(filename_mesh='meshes/dns_32x16.vtk',
           output_dir='output',
           n_step=10,
           n_maxiter=50,
           save_qp=False,
           load_mode='uniform',
           displ_val=[0.03, 0],
           eps_a=1e4,
           eps_r=1e-2,
           integration='full',  # reduced|full
           ):

    filename_mesh = osp.join(wdir, filename_mesh)

    mmesh = meshio.read(filename_mesh)
    x1 = mmesh.points[:, 0].max()

    options = {
        'nls': 'newton',
        'ls': 'ls',
        'ts': 'ts',
        'output_dir': output_dir,
        'mesh_update_variables': ['u'],
        'nls_iter_hook': ulf_iteration_hook,
        'pre_process_hook': ulf_init,
        'post_process_hook': post_process,
        'post_process_hook_final': post_process_final,
        'progress_filename': 'time_progress.csv',
    }

    functions = {
        'move': (move_fun,),
    }

    materials = {
        'solid': ({
            'K': {'Om': 5.7e9, 'Oc': 43.21e9},
            'mu': {'Om': 1.35e9, 'Oc': 28.46e9},
        },),
    }

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'U': ('parameter field', 'displacement', '(set-to-None)'),
    }

    regions = {
        'Omega': 'all',
        'Om': 'cells of group 1',
        'Oc': 'cells of group 2',
        'Left': ('vertices in (x < 0.001)', 'facet'),
        'Right': (f'vertices in (x > {x1 * 0.99})', 'facet'),
    }

    ebcs = {
        'l': ('Left', {'u.all': 0.0}),
        'r': ('Right', {'u.all': 'move'}),
    }

    if integration == 'reduced':
        # 3 point quadrature rule
        integrals = {'i': 1}
    else:
        # 4 point quadrature rule
        integrals = {'i': {'name': 'i', 'order': 3, 'full_order': True}}

    get_homog = get_hyperelastic_mat

    equations = {
        'balance': 'dw_ul_he_by_fun.i.Omega(get_homog, v, u) = 0',
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {
            'eps_a': eps_a,
            'eps_r': eps_r,
            'i_max': n_maxiter,
            'ls_on': 10.,
        }),
        'ts': ('ts.simple', {
            't0': 0,
            't1': 1,
            'n_step': n_step,
            'verbose': 1,
        })
    }

    return locals()
