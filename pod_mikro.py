# -*- coding: utf-8 -*-
import time
import numpy as nm
import numba as nb
import scipy.sparse as sps
import nonlinear_homogenization as nonlin_homog
from sfepy.base.base import output
from sfepy.base import multiproc

mp_module, _ = multiproc.get_multiproc()
basis_cache = mp_module.get_dict('basis_cache', clear=True)


@nb.njit('(float64[:,::1], float64[::1,:])')
def mymul_mat(matrices_a, matrices_b):
    m, o = matrices_a.shape
    n = matrices_b.shape[1]
    assert o == matrices_b.shape[0]

    result_matrices = nm.empty((m, n))
    for i in range(m):
        for j in range(n):
            result_matrices[i,j] = 0
            for k in range(o):
                result_matrices[i,j] += matrices_a[i,k] * matrices_b[k,j]

    return result_matrices


@nb.njit('(float64[:,::1], float64[::1])')
def mymul_vec(matrices_a, matrices_b):
    m, n = matrices_a.shape
    assert n == matrices_b.shape[0]

    result_matrices = nm.empty((m,))
    for i in range(m):
        result_matrices[i] = 0
        for j in range(n):
            result_matrices[i] += matrices_a[i,j] * matrices_b[j]

    return result_matrices


def get_transformation_basis(problem, T=False):
    if 'flag_all' not in basis_cache:
        basis_cache['flag_all'] = True
        basis_cache['lock'] = True

        print('>>> set_basis')
        for k, v in problem.conf.reduced_basis.items():
            if len(k[1]) > 0:
                basis_cache['flag_all'] = False

            basis_cache[k] = nm.ascontiguousarray(v)
            basis_cache[k + ('T',)] = nm.ascontiguousarray(v.T)

        basis_cache['lock'] = False

    corr_id = problem.homogen_corr_id

    if basis_cache['flag_all']:
        corr_id = (corr_id[0].split('|')[0], tuple())
    else:
        corr_id = (corr_id[0].split('|')[0],) + corr_id[1:]

    if basis_cache['lock']:
        while basis_cache['lock']:
            time.sleep(0.1)

    if T:
        return basis_cache[corr_id + ('T',)]
    else:
        return basis_cache[corr_id]


def mor_solution_fun(x, problem):
    return mymul_vec(problem.basis_cache, x)


def mor_system_fun(mtx, rhs, x0, problem):
    problem.basis_cache = basis = get_transformation_basis(problem)
    basisT = get_transformation_basis(problem, T=True)
    aux = nm.asfortranarray(nm.hstack([mtx @ basis, rhs[:, None], x0[:, None]]))
    auxr = mymul_mat(basisT, aux)

    return sps.csc_matrix(auxr[:, :-2]), auxr[:, -2], auxr[:, -1]


def line_search_fun(vec_x0, vec_r, vec_dx0, it, err_last, conf, fun,
                    aux, timers, log=None, context=None):
# def line_search_fun(vec_x0, vec_dx0, it, err_last, conf, fun,
#                     timers, log=None, context=None):
    vec_x = vec_x0 - vec_dx0

    timers.residual.start()
    try:
        vec_r = fun(vec_x)

    except ValueError:
        import ipdb; ipdb.set_trace()
        ok = False

    else:
        ok = True

    timers.residual.stop()

    svec_r = mymul_vec(get_transformation_basis(context, T=True), vec_r)

    err = nm.linalg.norm(svec_r)
    if not nm.isfinite(err):
        output('residual:', svec_r)
        output(nm.isfinite(svec_r).all())
        raise ValueError('infs or nans in the residual')

    if log is not None:
        log(err, it)

    return vec_x, vec_r, err, ok


def define(dim, filename_mesh, reduced_basis, **kwargs):
    d = nonlin_homog.define(dim, filename_mesh, **kwargs)

    d['requirements']['corrs_rs']['solvers'] = {
        'nls': 'nls_mor',
        'ls': 'ls_mor',
    }
    d['requirements']['corrs_rs']['ebcs'] = []
    d['requirements']['corrs_rs']['epbcs'] = []

    # d['options']['multiprocessing'] = False

    d['solvers'].update({
        'ls_mor': ('ls.scipy_direct', {'use_presolve': True}),
        'nls_mor': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-4,
            'problem': 'nonlinear',
            'scale_system_fun': mor_system_fun,
            'scale_solution_fun': mor_solution_fun,
            'line_search_fun': line_search_fun,
            'scaled_error': True,
        }),
    })

    d['reduced_basis'] = reduced_basis

    return d
