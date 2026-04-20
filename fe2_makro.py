"""
Two-scale nonlinear simulation - FE2 approach.

Running simulation: sfepy-run fe2_makro.py
"""
import os.path as osp
from functools import partial
import numpy as nm
from sfepy.base.base import Struct, output
from sfepy.discrete.evaluate import Evaluator
from sfepy.mechanics.tensors import get_full_indices, dim2sym
from sfepy.linalg.utils import invs_fast, dot_sequences
from sfepy.homogenization.micmac import get_homog_coefs_nonlinear

wdir = osp.dirname(__file__)

hyperelastic_data = {
    'micro': {
        'mtx_f': None,
        'coefs': {},
        'mat_eval_count': [],
        'to_save': {},
    },
    'macro': {
        'niter': [],
        'problem': None,
    },
}


def polar_decomposition(mtx_f):
    mtx_w, mtx_s, mtx_zt = nm.linalg.svd(mtx_f)
    mtx_r = nm.einsum('imn,ink->imk', mtx_w, mtx_zt, optimize=True)
    mtx_u = nm.einsum('inm,in,inl->iml', mtx_zt, mtx_s, mtx_zt, optimize=True)

    return mtx_r, mtx_u


tensor_idx_tab = {
    '2_3': [((0, 0), 0), ((1, 1), 1), ((1, 0), 2), ((0, 1), 2)],
    '2_4': [((0, 0), 0), ((0, 1), 1), ((1, 0), 2), ((1, 1), 3)],
    '3_6': [((0, 0), 0), ((1, 1), 1), ((2, 2), 2),
            ((1, 0), 3), ((0, 1), 3),
            ((2, 0), 4), ((0, 2), 4),
            ((2, 1), 5), ((1, 2), 5)],
    '3_9': [((0, 0), 0), ((0, 1), 1), ((0, 2), 2),
            ((1, 0), 3), ((1, 1), 4), ((1, 2), 5),
            ((2, 0), 6), ((2, 1), 7), ((2, 2), 8)],
}

def as_tensor(mtx, dim):
    d1, d2 = mtx.shape[-2:]
    nn = mtx.shape[:-2]

    idxs = tensor_idx_tab[f'{dim}_{d1}']
    nidx = tuple(slice(0, k) for k in nn)

    if d2 == 1:
        out = nm.empty(nn + (dim, dim), dtype=nm.float64)
        for idx, k in idxs:
            out[nidx + idx] = mtx[..., k, 0]

    elif d2 == d1:
        out = nm.empty(nn + (dim, dim, dim, dim), dtype=nm.float64)
        for idx1, k1 in idxs:
            for idx2, k2 in idxs:
                out[nidx + idx1 + idx2] = mtx[..., k1, k2]

    return out


def as_matrix(tensor, is_sym=False):
    n, dim = tensor.shape[:2]

    d = dim2sym(dim) if is_sym else dim**2
    idxs = {k: idx for idx, k in tensor_idx_tab[f'{dim}_{d}']}

    nidx = (slice(0, n),)
    if len(tensor.shape) == 3:
        out = nm.empty((n, d, 1), dtype=nm.float64)
        for k, idx in idxs.items():
            out[:, k, 0] = tensor[nidx + idx]

    elif len(tensor.shape) == 5:
        out = nm.empty((n, d, d), dtype=nm.float64)
        for k1, idx1 in idxs.items():
            for k2, idx2 in idxs.items():
                out[:, k1, k2] = tensor[nidx + idx1 + idx2]
    else:
        print(tensor.shape)
        raise ValueError()

    return out


def transform_R(mtx_r, obj):
    dim = mtx_r.shape[-1]
    dim2 = dim**2
    sym = dim2sym(dim)
    d1, d2 = obj.shape[-2:]

    if len(obj.shape) == 2:
        obj = obj[None, ...]

    if (d1 == sym) and (d2 == 1):
        out = dot_sequences(mtx_r, as_tensor(obj, dim), 'AB')
    elif (d1 == sym) and (d2 == sym):
        out = dot_sequences(dot_sequences(mtx_r, as_tensor(obj, dim), 'AB'),
                            mtx_r, 'ABT')
    elif (d1 == dim) and (d2 == 1):
        out = dot_sequences(mtx_r, obj, 'AB')
    elif (d1 == dim) and (d2 == dim):
        out = dot_sequences(mtx_r, obj, 'AB')
    elif (d1 == dim2) and (d2 == 1):
        if obj.shape[0] == 1:
            out = nm.einsum('zmi,znj,ij->zmn', mtx_r, mtx_r,
                            as_tensor(obj, dim)[0, ...], optimize=True)
        else:
            out = nm.einsum('zmi,znj,zij->zmn', mtx_r, mtx_r,
                            as_tensor(obj, dim), optimize=True)
        out = as_matrix(out)
    elif (d1 == dim2) and (d2 == dim2):
        if obj.shape[0] == 1:
            out = nm.einsum('zmi,znj,zok,zpl,ijkl->zmnop', mtx_r, mtx_r, mtx_r,
                            mtx_r, as_tensor(obj, dim)[0, ...], optimize=True)
        else:
            out = nm.einsum('zmi,znj,zok,zpl,zijkl->zmnop', mtx_r, mtx_r,
                            mtx_r, mtx_r, as_tensor(obj, dim), optimize=True)
        out = as_matrix(out)
    else:
        raise ValueError()

    return out


def get_sym(obj):
    dim = obj.shape[-1]
    obj = (obj + obj.transpose((0, 2, 1))) * 0.5
    obj = obj.reshape(-1, dim**2)

    if dim == 2:
        out = obj[:, nm.array([0, 3, 1])]
    elif dim == 3:
        out = obj[:, nm.array([0, 4, 8, 5, 2, 1])]
    else:
        raise ValueError('wrong dimension!')

    return out


def mat_rotate(values, mtx_r):
    out = {}
    for k, v in values.items():
        if k.startswith('s') or k.startswith('Volume'):
            continue

        if mtx_r is not None:
            aux = transform_R(mtx_r, v)
            if k == 'S':
                aux = get_sym(aux)[..., None]
        else:
            aux = v

        out[k] = nm.ascontiguousarray(aux)

    return out


def defgrad2strain(mtx_f):
    return mtx_f - nm.eye(mtx_f.shape[-1])


def strain2defgrad(mtx_e):
    dim = {3: 2, 6: 3}[mtx_e.shape[1]]
    full_idxs = get_full_indices(dim)
    return mtx_e[:, full_idxs] + nm.eye(dim)[None, ...]  # F = E + I


def get_rel_defgraf(mtx_f0, mtx_f):
    mtx_f0_inv = invs_fast(mtx_f0[None, ...])[0, ...]
    return dot_sequences(mtx_f, mtx_f0_inv, 'AB')


def compute_micro(mtx_f, micro_data, pb, ts, is_polar=None, define_args={}):
    if is_polar is None:
        is_polar = pb.conf.polar_decomposition

    dim = mtx_f.shape[-1]

    if is_polar:
        mtx_r, mtx_u = polar_decomposition(mtx_f)
    else:
        mtx_r, mtx_u = None, mtx_f

    mtx_f_prev = micro_data['mtx_f']

    output(f'>>> n_points={mtx_f.shape[0]}')
    if mtx_f_prev is not None:
        mtx_f_rel = get_rel_defgraf(mtx_f_prev, mtx_u)
    else:
        mtx_f_rel = mtx_u

    micro_data['mtx_f'] = mtx_u.copy()

    macro_data = {
        'mtx_e': defgrad2strain(mtx_f_rel),
        'mtx_r': mtx_r,
        'step_iter': (ts.step, pb.iiter),
    }

    define_args.update({
        'filename_mesh': pb.conf.filename_mesh_micro,
        'dim': dim,
        'multiprocessing': pb.conf.multi,
        'equilibrium_eps': pb.conf.micro_equilib_eps,
        'output_dir': pb.conf.output_dir,
    })

    out, deps = get_homog_coefs_nonlinear(ts, mtx_u, 'qp',
                                          macro_data=macro_data,
                                          define_args=define_args,
                                          problem=pb, iteration=pb.iiter,
                                          ret_corrs=True)

    micro_data['mat_eval_count'].append(mtx_f.shape[0])

    if is_polar:
        out = mat_rotate(out, mtx_r)

    return out, deps


def get_homog_mat(family_data, mode,
                  micro_fun=compute_micro,
                  micro_data=hyperelastic_data['micro'],
                  micro_idxs=None):

    pb = hyperelastic_data['macro']['problem']
    ts = pb.get_timestepper()
    output(f'>>> macro mat. fun: step={ts.step}, iiter={pb.iiter}')
    output(f'  micro_fun: {micro_fun.__name__}')

    ckey = (ts.step, pb.iiter)
    ccache = micro_data['coefs']
    n_el, n_qp, dim, _ = family_data.mtx_f.shape
    sym = family_data.green_strain.shape[2]
    dim2 = dim**2

    if ckey not in ccache:
        ccache.clear()

        def_args = {
            'recovery_idxs': [e * n_qp + q for e, q in pb.conf.recovery_idxs],
        }

        mtx_f = family_data.mtx_f.reshape((n_el * n_qp, dim, dim))
        ccache[ckey], corrs = micro_fun(mtx_f, micro_data, pb, ts,
                                        define_args=def_args)

        if pb.conf.save_qp:
            storage = micro_data['to_save']
            if len(storage) == 0:
                storage.update({'step_iter': [], 'mtx_f': [],
                                'A': [], 'S': [],
                                'micro_niter': [],
                                'micro_du': []})
            else:
                storage['step_iter'].append(ckey)
                storage['mtx_f'].append(mtx_f)
                storage['A'].append(ccache[ckey]['A'])
                storage['S'].append(ccache[ckey]['S'])
                if corrs is not None and 'equilibrium' in corrs:
                    niter = [cr.niter for cr in corrs['equilibrium']]
                    ndu = nm.array([nm.linalg.norm(cr.state) for cr in corrs['equilibrium']])

                    storage['micro_niter'].append(niter)
                    storage['micro_du'].append(nm.max(ndu))

    else:
        output('>>>  cached')

    coefs = ccache[ckey]

    if mode == 'tan_mod':
        out = coefs['A'].reshape((n_el, n_qp, dim2, dim2))
    elif mode == 'stress':
        out = coefs['S'].reshape((n_el, n_qp, sym, 1))
    else:
        raise ValueError()

    return out


def ulf_iteration_hook(pb, nls, vec, it, err, err0):
    Evaluator.new_ulf_iteration(pb, nls, vec, it, err, err0)
    pb.iiter = it
    with open(pb.conf.options.progress_filename, 'at') as f:
        ts = pb.get_timestepper()
        data = [ts.n_step, ts.step, it, err, err0]
        f.write(','.join([str(k) for k in data]) + '\n')
        f.flush()


def ulf_init(pb):
    pb.domain.mesh.coors_act = pb.domain.mesh.coors.copy()
    pb.iiter = 0

    hyperelastic_data['macro']['problem'] = pb
    with open(pb.conf.options.progress_filename, 'wt') as f:
        f.write(','.join(['nsteps', 'step', 'iter', 'err', 'err0']) + '\n')


def post_process_final(pb, state, out={}):
    from scipy.io import savemat

    micro_data = hyperelastic_data['micro']
    output(f'>>> num. mat. eval: {sum(micro_data["mat_eval_count"])}')
    # macro_data = hyperelastic_data['macro']
    # output(f'>>> num. iterations: {macro_data["niter"]}')
    # output(f'>>> total num. iterations: {nm.sum(macro_data["niter"])}')

    storage = micro_data['to_save']
    if len(storage) > 0:
        out.update({k: nm.array(v) for k, v in storage.items()})
        fname = osp.join(pb.conf.options['output_dir'], 'cf_qp.mat')
        savemat(fname, out)


def post_process(out, pb, state, extend=False):
    ev = partial(pb.evaluate, mode='el_avg', get_homog=pb.conf.get_homog)
    stress = ev('dw_ul_he_by_fun.i.Omega(get_homog, v, u)', term_mode='stress')
    strain = ev('dw_ul_he_by_fun.i.Omega(get_homog, v, u)', term_mode='strain')
    out['cauchy_stress'] = Struct(name='output_data', mode='cell', data=stress)
    out['green_strain'] = Struct(name='output_data', mode='cell', data=strain)
    vol = ev('ev_volume.i.Omega(u)')
    out['volume'] = Struct(name='output_data', mode='cell', data=vol)

    hyperelastic_data['macro']['niter'].append(pb.iiter)

    output('macro displacements:')
    displ = out['u'].data
    step = pb.ts.step
    for k, d0, in enumerate(displ):
        output(f'  nd={k}, step={step}: {d0}')

    return out


def load_fun(ts, coor, mode=None, problem=None, **kwargs):
    if mode == 'qp':
        if problem.conf.load_mode == 'uniform':
            mul = problem.conf.force * ts.nt
            val = nm.zeros((coor.shape[0], 2, 1), dtype=nm.float64)
            val[:, 0, 0] = coor[:, 1] / 0.2 * mul
        elif problem.conf.load_mode.startswith('ramp_'):
            ramp = float(problem.conf.load_mode.split('_')[1])
            cut = 1. if ts.nt > ramp else 1. / ramp
            mul = problem.conf.force * ts.nt * cut
            val = nm.zeros((coor.shape[0], 2, 1), dtype=nm.float64)
            val[:, 0, 0] = coor[:, 1] / 0.2 * mul
        else:
            import pdb; pdb.set_trace()

        output(f'>>> load: {mul}')

        return {'val': val}


def get_boundary(coors, domain=None, side='right', eps=1e-2):
    xn = coors[:, 0] - coors[:, 0].min()
    xn /= xn.max()

    if side == 'right':
        return nm.where(xn > (1 - eps))[0]
    elif side == 'left':
        return nm.where(xn < eps)[0]
    else:
        return None


def define(filename_mesh='meshes/macro_L.vtk',
           filename_mesh_micro='meshes/micro_1.vtk',
           output_dir='output',
           n_step=10,
           n_maxiter=50,
           save_qp=False,
           polar_decomposition=True,
           force=1.0e8,
           multi=True,
           micro_equilib_eps=None,
           load_mode='uniform',  # uniform|ramp_
           recovery_idxs=[],
           eps_a=1e3,
           eps_r=1e-3,
           integration='full',  # reduced|full
           ):

    filename_mesh = osp.join(wdir, filename_mesh)
    filename_mesh_micro = osp.join(wdir, filename_mesh_micro)

    options = {
        'output_dir': output_dir,
        'mesh_update_variables': ['u'],
        'nls_iter_hook': ulf_iteration_hook,
        'pre_process_hook': ulf_init,
        'post_process_hook': post_process,
        'post_process_hook_final': post_process_final,
        'micro_filename': osp.join(wdir, 'fe2_mikro.py'),
        'progress_filename': 'time_progress.csv',
    }

    functions = {
        'load_fun': (load_fun,),
        'get_left': (partial(get_boundary, side='left'),),
        'get_right': (partial(get_boundary, side='right'),),
    }

    materials = {
        'load' : (None, 'load_fun'),
    }

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
    }

    regions = {
        'Omega': 'all',
        'Left': ('vertices by get_left', 'facet'),
        'Right': ('vertices by get_right', 'facet'),
    }

    ebcs = {
        'l': ('Left', {'u.all': 0.0}),
    }

    if integration == 'reduced':
        # 3 point quadrature rule
        integrals = {'i': 1}
    else:
        # 4 point quadrature rule
        integrals = {'i': {'name': 'i', 'order': 3, 'full_order': True}}

    get_homog = partial(get_homog_mat, micro_fun=compute_micro)

    equations = {
        'balance':
            'dw_ul_he_by_fun.i.Omega(get_homog, v, u) = dw_surface_ltr.i.Right(load.val, v)'
    }

    solvers = {
        'ls': ('ls.mumps', {}),
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
        })
    }

    return locals()
