"""
Two-scale nonlinear simulation - POD based complexity reduction
"""
import os.path as osp
import time
from functools import partial
import numpy as nm
from scipy.io import loadmat, savemat
from csa_makro import merge_locals
from fe2_makro import (hyperelastic_data, compute_micro,
                       get_homog_mat, ulf_init, strain2defgrad)
import fe2_makro as nonlin_macro
from sfepy.base.base import output
from sfepy.solvers.ts import TimeStepper
from sfepy.mechanics.tensors import dim2sym

wdir = osp.dirname(__file__)


def compute_micro_basis(mtx_u, micro_data, pb, ts):
    _, corrs = compute_micro(mtx_u, micro_data, pb, ts, is_polar=True)

    return {k: v for k, v in corrs.items() if k[:4] == 'corr'}


# bounds: [(e_comp, val), ...]
def compute_micro_database(pb, bounds, n):
    dim = pb.domain.mesh.dim
    ts = TimeStepper(t0=0, t1=1, n_step=n)

    npar = len(bounds)
    e_bounds = nm.zeros((npar, dim2sym(dim)), dtype=nm.float64)

    for k, (comp, val) in enumerate(bounds):
        e_bounds[k, comp] = val

    micro_data = hyperelastic_data['micro']

    corr_comps = []
    corr_ts = {}

    for step, itime in ts:
        mtx_u = strain2defgrad(e_bounds * itime)
        corrs = compute_micro_basis(mtx_u, micro_data, pb, ts)
        for k, v in corrs.items():
            if step == 0:
                corr_ts[k] = [[] for k in range(npar)]
                corr_comps += [(k, comp) for comp in v[0].components]
            for ii, iv in enumerate(v):
                corr_ts[k][ii].append(iv)

    return corr_ts, corr_comps


def reduced_basis(pb, bounds, nsteps, nmax=100, mode='comp'):
    def esolve_scipy(mtx, n):
        import scipy.linalg as sla

        d = mtx.shape[0]
        evals, evecs = sla.eigh(mtx, driver='evx',
                                subset_by_index=[d - n, d - 1])

        return evals, evecs

    def compute_basis(mtx_U, nmax, ckey, ccomp):
        mtx_V = nm.hstack(mtx_U)
        mtx_Q = nm.dot(mtx_V, mtx_V.T)
        mem = f'{mtx_Q.nbytes / (2**30):.2}GB'
        output(f'>>> {ckey}/{ccomp}:')
        output(f'  calculating eigenvalues of covariance matrix {mtx_Q.shape},'
               f' nmax={nmax}, memory={mem}')

        t0 = time.time()
        evals, evecs = esolve_scipy(mtx_Q, nmax)
        dt = time.time() - t0
        output(f'  calculated in {dt:.2f}s')

        return {(ckey, ccomp): evecs, ('e_' + ckey, ccomp): evals}

    corrs, corr_comps = compute_micro_database(pb, bounds, nsteps)
    basis = {}

    mtx_U = []
    for ckey, ccomp in corr_comps:
        cvals = corrs[ckey]

        for cval in cvals:
            vecs = nm.hstack([k.states[ccomp]['u'][:, None] for k in cval])
            vecs0 = 0
            mtx_U.append(vecs - vecs0)

        if mode == 'comp':
            basis.update(compute_basis(mtx_U, nmax, ckey, ccomp))
            mtx_U = []

    if mode == 'all':
        basis.update(compute_basis(mtx_U, nmax, ckey, tuple()))

    return basis


def compute_micro_pod(mtx_f, micro_data, pb, ts, define_args={}):
    t0 = time.time()

    define_args.update({'reduced_basis': hyperelastic_data['reduced_basis']})
    out = compute_micro(mtx_f, micro_data, pb, ts,
                        is_polar=True, define_args=define_args)

    print(f'>>> micro: n_micro={mtx_f.shape}, dt={time.time() - t0}')

    return out


def ulf_init_pod(pb):
    def get_key(key):
        skey = key.split('|')
        return (skey[0], tuple(int(k) for k in skey[1:]))

    ulf_init(pb)

    conf = pb.conf

    bounds = [
        # e_11
        (0, -0.5),
        (0, 0.5),
        # e_22
        (1, -0.5),
        (1, 0.5),
        # e_12
        (2, -0.5),
        (2, 0.5),
    ]

    nsteps = 10

    basis_key = 'x'.join([f'{k[0]}_{k[1]}' for k in bounds]) + f'_{nsteps}'
    basis_key = basis_key + f'_{conf.mode}'

    fname_micro_mesh = conf.filename_mesh_micro
    fname_micro_mesh = osp.splitext(osp.split(fname_micro_mesh)[-1])[0]

    fname_basis = f'reduced_bais_{fname_micro_mesh}_{basis_key}X.mat' #!!!!!!!!!!!!!!
    if conf.basis_dir is not None:
        fname_basis = osp.join(conf.basis_dir, fname_basis)
    else:
        fname_basis = osp.join(pb.output_dir, fname_basis)

    if osp.isfile(fname_basis):
        basis_ = loadmat(fname_basis)
        ctime = basis_.pop('ctime', None)
        basis = {get_key(ckey): cval for ckey, cval in basis_.items()
                 if ckey[:2] != '__'}
    else:
        micro_filename = conf.options.micro_filename
        conf.options.micro_filename = conf.micro_basis_filename

        t0 = time.time()
        basis = reduced_basis(pb, bounds, nsteps, conf.nmax, mode=conf.mode)
        basis_ = {f'{"|".join([ckey] + [str(k) for k in ccomp])}': cval
                 for (ckey, ccomp), cval in basis.items()}
        basis_['ctime'] = time.time() - t0
        savemat(fname_basis, basis_)

        # reset micro data
        hyperelastic_data['micro'].update({
            'mtx_f': None,
            'mat_eval_count': [],
            'states': {},
        })

        conf.options.micro_filename = micro_filename

        del pb.homogen_app

    rbasis = {}
    for (ckey, ccomp), cval in basis.items():
        if ckey.startswith('e_'):
            continue

        evals = basis[('e_' + ckey, ccomp)]
        evecs = basis[(ckey, ccomp)]

        esum = nm.cumsum(evals) / evals.sum()
        idxs = nm.where(esum >= conf.delta**2)[0]
        output(f'  reduced basis dimension: {len(idxs)}')

        if len(idxs) == conf.nmax:
            raise ValueError('nred = nmax!')

        rbasis[(ckey, ccomp)] = evecs[:, idxs[::-1]]

    hyperelastic_data['reduced_basis'] = rbasis


def define(delta=1e-3,
           err_idxs=[],
           nmax=250,
           mode='all',
           basis_dir=None,
           **kwargs):

    print(f'>>> delta: {delta}')
    print(f'>>> err_idxs: {err_idxs}')

    d = nonlin_macro.define(**kwargs)

    micro_basis_filename = 'fe2_mikro.py'
    micro_filename = 'pod_mikro.py'

    options = {
        'micro_filename': osp.join(wdir, micro_filename),
        'pre_process_hook': ulf_init_pod,
        'recover_micro': False,
    }

    get_homog_pod = partial(get_homog_mat, micro_fun=compute_micro_pod)

    equations = {
        'balance': """dw_ul_he_by_fun.i.Omega(get_homog_pod, v, u)
                    = dw_surface_ltr.i.Right(load.val, v)""",
    }

    return merge_locals(locals(), d)
