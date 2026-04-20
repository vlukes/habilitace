"""
Two-scale nonlinear simulation - CSA reduction algorithm.
"""
import numpy as nm
import os.path as osp
import fe2_makro as nonlin_macro
from fe2_makro import (defgrad2strain, get_rel_defgraf, strain2defgrad,
                       polar_decomposition, transform_R, mat_rotate,
                       get_sym, hyperelastic_data, get_homog_mat)
from sfepy.base.base import Struct, output
from sfepy.linalg.utils import invs_fast, dot_sequences
import sfepy.base.multiproc as multi
from sfepy.homogenization.utils import iter_sym

wdir = osp.dirname(__file__)

hyperelastic_data['micro'].update({
    'mtx_e': None,
    'states': {},
    'ccoefs': {},
})

hyperelastic_data['micro_check'] = {
    'mtx_e': None,
    'states': {},
    'coefs_chck': {},
    'coefs_approx': {},
    'ckeys': [],
    'mat_eval_count': [],
}

hyperelastic_data['n_maxiter'] = None
hyperelastic_data['n_new_centroids'] = []


def post_process_final(pb, state):
    from scipy.io import savemat

    micro_data = hyperelastic_data['micro']

    out = {
        'ncentroids': micro_data["mat_eval_count"],
    }

    nonlin_macro.post_process_final(pb, state, out)

    if len(pb.conf.err_idxs) > 0:
        micro_data_chck = hyperelastic_data['micro_check']

        output('>>> Micro check:')
        output(f'>>> num. mat. eval: {sum(micro_data_chck["mat_eval_count"])}')

        out_chck = {k: nm.array(v).transpose((1, 0, 2, 3))
                    for k, v in micro_data_chck['coefs_chck'].items()}

        out_approx = {k + '_approx': nm.array(v).transpose((1, 0, 2, 3))
                    for k, v in micro_data_chck['coefs_approx'].items()}

        out_ext = {
            'ncheck': micro_data_chck["mat_eval_count"],
            'ckeys': micro_data_chck['ckeys'],
        }

        fname = osp.join(pb.conf.options['output_dir'],
                        f'cf_check_{pb.conf.delta}.mat')
        savemat(fname, {**out_chck, **out_approx, **out_ext})


def post_process(out, pb, state, extend=False):
    out = nonlin_macro.post_process(out, pb, state, extend)

    cid = hyperelastic_data['cid']
    ncid = hyperelastic_data['ncid']
    n_el = pb.domain.mesh.n_el
    cid = cid.reshape((n_el, -1))
    ncid = ncid.reshape((n_el, -1))
    for k in range(cid.shape[1]):
        out[f'cid{k}'] = Struct(name='output_data', mode='cell',
                                data=cid[:, [k], None, None])
        out[f'ncid{k}'] = Struct(name='output_data', mode='cell',
                                 data=ncid[:, [k], None, None])

    output('macro displacements:')
    displ = out['u'].data
    step = pb.ts.step
    for k, d0, in enumerate(displ):
        output(f'  nd={k}, step={step}: {d0}')

    return out


def get_homogen_app(problem, n_micro, define_args=None, output_dir=None):
    from sfepy.base.conf import ProblemConf, get_standard_keywords
    from sfepy.homogenization.homogen_app import HomogenizationApp

    if not hasattr(problem, 'homogen_app'):
        required, other = get_standard_keywords()
        required.remove('equations')
        micro_file = problem.conf.options.micro_filename
        conf = ProblemConf.from_file(micro_file, required, other,
                                     verbose=False, define_args=define_args)
        if output_dir is not None:
            conf.options.output_dir = output_dir
        options = Struct(output_filename_trunk=None)
        app = HomogenizationApp(conf, options, 'micro:', n_micro=n_micro)
        problem.homogen_app = app

        if hasattr(app.app_options, 'use_mpi') and app.app_options.use_mpi:
            multiproc, multiproc_mode = multi.get_multiproc(mpi=True)
            multi_mpi = multiproc if multiproc_mode == 'mpi' else None
        else:
            multi_mpi = None

        if multi_mpi is not None:
            multi_mpi.master_send_task('init', (micro_file, n_micro))

        app.multi_mpi = multi_mpi
    else:
        app = problem.homogen_app

    return app


def get_homog_coefs_nonlinear_csa(ts, macro_data, problem, define_args,
                                  micro_states):
    oprefix = output.prefix
    output.prefix = 'micro:'

    n_micro = macro_data['mtx_e'].shape[0]
    app = get_homogen_app(problem, n_micro=n_micro, define_args=define_args)

    if micro_states:
        app.n_micro = n_micro
        app.micro_states = {'coors': micro_states['coors'].copy()}
        for k, v in micro_states['corrs'].items():
            app.updating_corrs[k] = v

    else:
        app.n_micro = n_micro
        act_coors = app.problem.get_mesh_coors(actual=False)
        app.micro_states = {'coors': nm.tile(act_coors, (n_micro, 1, 1))}
        app.updating_corrs = None

    if macro_data is not None:
        macro_data['macro_time_step'] = ts.step

    app.setup_macro_data(macro_data)

    if app.multi_mpi is not None:
        app.multi_mpi.master_send_task('calculate',
                                       (macro_data, ts, problem.iiter))

    coefs, deps = app(ret_all=True, itime=ts.step, iiter=problem.iiter)

    if isinstance(coefs, tuple):
        coefs = coefs[0]

    out = {}
    for key, val in coefs.__dict__.items():
        if isinstance(val, list):
            out[key] = nm.array(val)
        elif isinstance(val, dict):
            for key2, val2 in val.items():
                out[key+'_'+key2] = nm.array(val2)

    for key in out:
        shape = out[key].shape
        if len(shape) == 1:
            out[key] = out[key].reshape(shape + (1, 1))
        elif len(shape) == 2:
            out[key] = out[key].reshape(shape + (1,))

    micro_states['coors'] = app.micro_states['coors'].copy()
    micro_states['corrs'] = {k: v.copy()
                             for k, v in  app.updating_corrs.items()}

    output.prefix = oprefix

    return out, deps, micro_states


def get_dist(strain):
    return nm.linalg.norm(strain, axis=1)


def get_dist_matrix(strain, strain0):
    d = strain[:, None, :] - strain0[None, :, :]
    sh = d.shape
    return get_dist(d.reshape((-1, sh[-1]))).reshape(sh[:-1])


def kmeans(mtx_e, centroids, maxiter=1000, tol=1e-6):
    npts, dim = mtx_e.shape
    ncent = centroids.shape[0]

    for iiter in range(maxiter):
        centroids_old = centroids.copy()

        dist = get_dist_matrix(mtx_e, centroids)
        clusters = nm.argmin(dist, axis=1)
        flag = nm.zeros((npts, ncent, dim), dtype=nm.float64)
        flag[nm.arange(npts), clusters, :] = mtx_e

        centroids = flag.sum(axis=0)
        flag = nm.zeros((npts, ncent), dtype=nm.int64)
        flag[nm.arange(npts), clusters] = 1
        nincent = flag.sum(axis=0)

        idxs = nincent > 0

        centroids[idxs, :] /= nincent[idxs, None]

        if nm.all(get_dist(centroids - centroids_old) <= tol):
            break

    if iiter == maxiter:
        print(' reached the maximum nuber of iterations!')

    return centroids, clusters, dist


def get_new_centroids(mtx_e, delta):
    npts = mtx_e.shape[0]
    centroids = nm.empty_like(mtx_e)
    centroids[0] = mtx_e[0]

    if npts > 1:
        k = 1
        while k <= npts:
            centroids[:k], idxs, dist = kmeans(mtx_e, centroids[:k])

            dist0 = nm.min(dist, axis=1)
            idx = dist0 > delta
            if not nm.any(idx):
                break

            centroids[k] = mtx_e[[nm.argmax(idx)]]
            k += 1

        centroids = centroids[:k]
        cmap = [nm.nonzero(idxs == k)[0] for k in range(centroids.shape[0])]
        idxs = nm.array([len(cm) > 0 for cm in cmap])
        cmap = [cm for (cm, i) in zip(cmap, idxs) if i]

        return centroids[idxs], cmap, dist[:, idxs]

    return centroids, [nm.array([0])], nm.array([[0]])


def get_weights(dist, delta):
    idxs = dist < delta
    weights = (delta - dist.copy())**2
    weights[nm.logical_not(idxs)] = 0
    wsum = weights.sum(axis=1)
    widxs = wsum > 0

    if nm.any(widxs):
        weights[widxs] /= wsum[widxs, None]

    return weights


def approximate_mat_interp(values, mtx_r, weights, sa=None):
    npts, ncen = weights.shape
    dim = mtx_r.shape[-1]

    out = {}
    for k, v in values.items():
        if k.startswith('s'):
            continue

        outc = nm.zeros((npts, ncen) + v.shape[1:], dtype=v.dtype)

        for icen, col in enumerate(weights.T):
            inside = col > 0
            if inside.any():
                val = v[icen]
                if sa is not None:
                    cen_f_inv, mtx_f = sa
                    rel_f = dot_sequences(mtx_f[inside],
                                          cen_f_inv[icen][None, ...], 'AB')
                    rel_e = defgrad2strain(rel_f)
                    val = nm.tile(val, (inside.sum(),) + (1,) * val.ndim)
                    for kk, irc in enumerate(iter_sym(dim)):
                        scf = values[f's{k}'][[icen], kk]
                        val += rel_e[:, irc[0], irc[1]][:, None, None] * scf

                aux = transform_R(mtx_r[inside], val)
                if k == 'S':
                    aux = get_sym(aux)[..., None]

                outc[inside, icen, ...] = aux

        out[k] = (outc * weights[..., None, None]).sum(axis=1)

    return out


def create_pis(coor):
    dim = coor.shape[1]
    pis = nm.zeros((dim, dim), dtype=object)
    for ir in range(dim):
        for ic in range(dim):
            pi = nm.zeros_like(coor)
            pi[:, ir] = coor[:, ic]
            pi.shape = (pi.shape[0] * pi.shape[1],)

            pis[ir,ic] = pi

    return pis


def approximate_geom_interp(ridxs, rvalues, mtx_r, weights, sa=None):
    npts, ncen = weights.shape
    dim = mtx_r.shape[-1]

    rec = nm.zeros((npts), dtype=bool)
    rec[ridxs] = True
    ridxs2 = nm.zeros((npts), dtype=nm.int64)
    ridxs2[rec] = nm.arange(len(ridxs))
    nmcoors = rvalues['coors'].shape[1]
    outrc = nm.zeros((len(ridxs), ncen, nmcoors, dim), dtype=nm.float64)

    for icen, col in enumerate(weights.T):
        inside = nm.logical_and(col > 0, rec)
        if inside.any():
            rcoor = rvalues['coors'][icen]
            rpi = create_pis(rcoor)
            rcorr = rvalues['corrs']['corrs_rs'][icen]

            if sa is not None:
                cen_f_inv, mtx_f = sa
                rel_f = dot_sequences(mtx_f[inside],
                                      cen_f_inv[icen][None, ...], 'AB')
                rel_e = defgrad2strain(rel_f)
                rcoor = nm.tile(rcoor, (inside.sum(),) + (1,) * rcoor.ndim)
                for ir, ic in iter_sym(dim):
                    rmul = (rcorr.states[ir, ic]['u'] + rpi[ir, ic])
                    rmul = rmul.reshape((-1, 2))
                    rcoor += rel_e[:, ir, ic][:, None, None] * rmul

            aux = dot_sequences(rcoor, mtx_r[inside], 'ABT')
            outrc[ridxs2[inside], icen, ...] = aux

    outr = (outrc * weights[ridxs, :, None, None]).sum(axis=1)

    return outr


def get_micro_states(saved_states, idxs):
    if saved_states:
        out = {}
        out['coors'] = saved_states['coors'][idxs].copy()
        out['corrs'] = {k: [v[i] for i in idxs]
                        for k, v in saved_states['corrs'].items()}

        return out
    else:
        return {}


def append_micro_states(saved_states, states):
    if saved_states:
        saved_states['coors'] = nm.append(saved_states['coors'],
                                          states['coors'], axis=0)
        for k, v in states['corrs'].items():
            saved_states['corrs'][k] += v
    else:
        saved_states['coors'] = states['coors'].copy()
        saved_states['corrs'] = {k: v for k, v in states['corrs'].items()}


def compute_micro(mtx_f, mtx_f0, states, pb, ts,
                  define_args={}, macro_flags=None):
    dim = mtx_f.shape[-1]

    output(f'  n_points={mtx_f.shape[0]}')

    if mtx_f0 is not None:
        # relF = F * inv(F0)
        mtx_f_rel = get_rel_defgraf(mtx_f0, mtx_f)
    else:
        mtx_f_rel = mtx_f

    macro_data = {'mtx_e': defgrad2strain(mtx_f_rel)}

    if macro_flags is not None:
        macro_data.update(macro_flags)

    define_args.update({
        'filename_mesh': pb.conf.filename_mesh_micro,
        'dim': dim,
        'multiprocessing': pb.conf.multi,
        'equilibrium_eps': pb.conf.micro_equilib_eps,
        'output_dir': pb.conf.output_dir,
    })

    out, deps, states = get_homog_coefs_nonlinear_csa(ts, macro_data, pb, 
                                                      define_args, states)

    nsym = out['S'].shape[1]
    if pb.conf.is_sa:
        divV = out['divV']

        sA0 = out['sAS'][..., :-1]
        sS = out['sAS'][..., :nsym, [-1]]

        # -c.S * c.divV[:, None] + c.sS1
        out['sS'] = -out['S'][:, None, :, :] * divV[:, :, None, :] + sS

        # -c.A * c.divV + c.sA0 + c.sA1 + c.sA2 + c.sA2.T
        out['sA'] = (-out['A'][:, None, :, :] * divV[:, :, None, :]
                    + sA0 + out['sA1']
                    + out['sA2'] + out['sA2'].transpose(0, 1, 3, 2))

        coefs = {k: v for k, v in out.items() if k in ['S', 'sS', 'A', 'sA']}
    else:
        coefs = {k: v for k, v in out.items() if k in ['S', 'A']}

    return coefs, deps, states


def compute_micro_approx(mtx_f, micro_data, pb, ts, define_args={}):
    n_macro = mtx_f.shape[0]
    centroids = micro_data['mtx_e']
    delta = pb.conf.delta
    ccoefs = micro_data['ccoefs']

    mtx_r, mtx_u = polar_decomposition(mtx_f)
    mtx_e = get_sym(defgrad2strain(mtx_u))

    if centroids is None:
        dist = nm.ones((n_macro, 1), dtype=nm.float64) * 1e12
    else:
        dist = get_dist_matrix(mtx_e, centroids)

    inside = nm.min(dist, axis=1) <= delta
    outside = nm.logical_not(inside)

    recovery_idxs = define_args.pop('recovery_idxs', [])

    if nm.any(outside) > 0:
        new_centroids, cmap, dist = get_new_centroids(mtx_e[outside, :], delta)

        mtx_u_outside = mtx_u[outside, :]
        mtx_u_cen = nm.array([nm.mean(mtx_u_outside[lcmap], axis=0)
                              for lcmap in cmap])

        output(f'  new_centroids={new_centroids.shape[0]}')

        if centroids is None:
            rel_cen = None
            mtx_u0_cen = None
        else:
            cdist = get_dist_matrix(new_centroids, centroids)
            rel_cen = nm.argmin(cdist, axis=1)
            mtx_u0_cen = strain2defgrad(centroids[rel_cen])

        states = get_micro_states(micro_data['states'], rel_cen)
        coefs_, deps, states = compute_micro(mtx_u_cen, mtx_u0_cen, states, pb, ts,
                                             define_args=define_args)
        append_micro_states(micro_data['states'], states)
        micro_data['mat_eval_count'].append(new_centroids.shape[0])

        for k, v in coefs_.items():
            if k not in ccoefs:
                ccoefs[k] = v.copy()
            else:
                ccoefs[k] = nm.append(ccoefs[k], v, axis=0)

        if centroids is None:
            centroids = new_centroids.copy()
        else:
            centroids = nm.append(centroids, new_centroids, axis=0)

        micro_data['mtx_e'] = centroids

    else:
        micro_data['mat_eval_count'].append(0)
        deps = None

    dist = get_dist_matrix(mtx_e, centroids)
    weights = get_weights(dist, delta)

    if pb.conf.is_sa:
        cen_f_inv = invs_fast(strain2defgrad(centroids)[None, ...])[0, ...]
        sa_data = (cen_f_inv, strain2defgrad(mtx_e))
    else:
        sa_data = None

    out = approximate_mat_interp(ccoefs, mtx_r, weights, sa=sa_data)

    if len(recovery_idxs) > 0 and ts.step > 0:
        hpb = pb.homogen_app.problem

        outr = approximate_geom_interp(recovery_idxs, micro_data['states'],
                                       mtx_r, weights, sa=sa_data)

        output_dir = hpb.conf.options.get('output_dir', '.')
        extra = f'recovered_{pb.ts.step:03d}_{pb.iiter:03d}'
        # extra = f'recovered_{ts.step:03d}'
        micro_name = hpb.get_output_name(extra=extra)
        filename = osp.join(output_dir, osp.basename(micro_name))

        hout = {}
        coors0 = hpb.domain.cmesh.coors
        for coor, idx in zip(outr, recovery_idxs):
            flag = f'_{idx}'
            hout[f'displacement{flag}'] = Struct(name='output_data',
                                                 mode='vertex',
                                                 data=coor - coors0)

        hpb.save_state(filename, out=hout)
        print(f'recovery file: {filename}')

        # def save_micro_state(pb, file_tag, displ, strain, stress, mtx_r):
        #     oStruct = partial(Struct, name='output_data', dofs=None)
        #     out = {}

        #     if mtx_r is None:
        #         flag = ''
        #     else:
        #         flag = '_no_rot'
        #         coors0 = pb.domain.get_mesh_coors()
        #         coors1 = coors0.copy()
        #         coors1 += displ
        #         coors1 = nm.dot(coors1, mtx_r.T)
        #         displ = coors1 - coors0

        #     out['cauchy_stress' + flag] = oStruct(mode='cell', data=stress)
        #     out['green_strain' + flag] = oStruct(mode='cell', data=strain)
        #     out['displacement'] = oStruct(mode='vertex', data=displ)

        #     output_dir = pb.conf.options.get('output_dir', '.')
        #     micro_name = pb.get_output_name(extra=f'recovered_{file_tag}')
        #     filename = osp.join(output_dir, osp.basename(micro_name))
        #     pb.save_state(filename, out=out)

    hyperelastic_data['cid'] = nm.argmin(dist, axis=1)
    hyperelastic_data['ncid'] = (weights > 0.).sum(axis=1)

    return out, deps


def compute_micro_direct(mtx_f, micro_data, pb, ts,
                         is_polar=True, define_args={}):
    n_macro = mtx_f.shape[0]
    centroids = micro_data['mtx_e']
    coefs = micro_data['coefs']

    if is_polar:
        mtx_r, mtx_u_cen = polar_decomposition(mtx_f)
    else:
        mtx_r, mtx_u_cen = None, mtx_f

    if centroids is None:
        rel_cen = None
        mtx_u0_cen = None
    else:
        rel_cen = nm.arange(n_macro) + centroids.shape[0] - n_macro
        mtx_u0_cen = centroids[rel_cen]

    states = get_micro_states(micro_data['states'], rel_cen)
    coefs_, deps, states = compute_micro(mtx_u_cen, mtx_u0_cen, states, pb, ts,
                                         define_args=define_args)
    append_micro_states(micro_data['states'], states)
    micro_data['mat_eval_count'].append(n_macro)

    for k, v in coefs_.items():
        if k not in coefs:
            coefs[k] = v.copy()
        else:
            coefs[k] = nm.append(coefs[k], v, axis=0)

    if centroids is None:
        micro_data['mtx_e'] = mtx_u_cen
    else:
        micro_data['mtx_e'] = nm.append(centroids, mtx_u_cen, axis=0)

    hyperelastic_data['cid'] = nm.arange(n_macro)
    hyperelastic_data['ncid'] = nm.ones((n_macro,), dtype=nm.int64)

    out = mat_rotate(coefs_, mtx_r)

    return out, deps


def get_homog_mat_clusters(family_data, mode):
    pb = hyperelastic_data['macro']['problem']
    micro_data = hyperelastic_data['micro']

    if pb.conf.delta is not None:
        micro_fun = compute_micro_approx
    else:
        micro_fun = compute_micro_direct

    coefs = get_homog_mat(family_data, mode, micro_fun, micro_data)

    if hyperelastic_data['n_maxiter'] is not None:
        nls = pb.get_solver().nls
        nls.conf.i_max = hyperelastic_data['n_maxiter']
        hyperelastic_data['n_maxiter'] = None

    if sum(micro_data['mat_eval_count'][-pb.conf.n_maxiter:]) == 0:
        nls = pb.get_solver().nls
        hyperelastic_data['n_maxiter'] = nls.conf.i_max
        nls.conf.i_max = pb.conf.n_maxiter

    # compute given microsctructures to check the above approximation
    if len(pb.conf.err_idxs) > 0:
        coefs0 = micro_data['coefs'][(pb.get_timestepper().step, pb.iiter)]
        check_micro(family_data, pb.conf.err_idxs, coefs0)

    return coefs


def compute_micro_directX(mtx_f, micro_data, pb, ts, is_fd=False):
    n_macro = mtx_f.shape[0]
    centroids = micro_data['mtx_e']
    mtx_r, mtx_u_cen = polar_decomposition(mtx_f)

    if centroids is None:
        rel_cen = None
        mtx_u0_cen = None
    else:
        rel_cen = nm.arange(n_macro) + centroids.shape[0] - n_macro
        mtx_u0_cen = centroids[rel_cen]

    states = get_micro_states(micro_data['states'], rel_cen)
    coefs_, deps, states = compute_micro(mtx_u_cen, mtx_u0_cen, states, pb, ts)

    if not is_fd:
        append_micro_states(micro_data['states'], states)
        micro_data['mat_eval_count'].append(n_macro)

        if centroids is None:
            micro_data['mtx_e'] = mtx_u_cen
        else:
            micro_data['mtx_e'] = nm.append(centroids, mtx_u_cen, axis=0)

    return mat_rotate(coefs_, mtx_r)


def check_micro(family_data, micro_idxs, coefs0):
    micro_data=hyperelastic_data['micro_check']

    pb = hyperelastic_data['macro']['problem']
    ts = pb.get_timestepper()
    output(f'>>> macro check: step={ts.step}, iiter={pb.iiter}')

    ckey = (ts.step, pb.iiter)

    if ckey not in micro_data['ckeys']:
        n_el, n_qp, dim, _ = family_data.mtx_f.shape

        micro_idxs = nm.array(micro_idxs)
        micro_idxs = micro_idxs[:, 0] * n_qp + micro_idxs[:, 1]

        micro_data['ckeys'].append(ckey)
        mtx_f = family_data.mtx_f.reshape((n_el * n_qp, dim, dim))[micro_idxs]
        coefs = compute_micro_directX(mtx_f, micro_data, pb, ts)

        coefs_chck = micro_data['coefs_chck']
        coefs_approx = micro_data['coefs_approx']

        for k in ['A', 'S']:
            if k not in coefs_chck:
                coefs_chck[k] = [coefs[k].copy()]
                coefs_approx[k] = [coefs0[k][micro_idxs].copy()]
            else:
                coefs_chck[k].append(coefs[k])
                coefs_approx[k].append(coefs0[k][micro_idxs])
    else:
        output('>>>  cached')


def merge_locals(dlocals, d):
    out = {}
    dkeys = list(d.keys())
    for k, v in dlocals.items():
        if k in d:
            if isinstance(v, dict) and isinstance(d[k], dict):
                d[k].update(v)
                out[k] = d[k]
            elif isinstance(v, list) and isinstance(d[k], list):
                out[k] = v + d[k]
            elif isinstance(v, tuple) and isinstance(d[k], tuple):
                out[k] = v + d[k]
            else:
                out[k] = v

            dkeys.remove(k)
        else:
            out[k] = v

    out.update({k: d[k] for k in dkeys})

    return out

def define(delta=0.005,
           err_idxs=[],
           is_sa=True,
           is_fd=False,
           n_maxiter=25,
           eps_a=1e4,
           **kwargs):

    print(f'>>> delta: {delta}')
    print(f'>>> err_idxs: {err_idxs}')

    d = nonlin_macro.define(n_maxiter=n_maxiter, eps_a=eps_a, **kwargs)

    if delta is None:
        is_sa = False

    micro_filename = 'csa_mikro.py'

    options = {
        'micro_filename': osp.join(wdir, micro_filename),
        'post_process_hook': post_process,
        'post_process_hook_final': post_process_final,
    }

    get_homog_clusters = get_homog_mat_clusters

    equations = {
        'balance': """dw_ul_he_by_fun.i.Omega(get_homog_clusters, v, u)
                    = dw_surface_ltr.i.Right(load.val, v)""",
    }

    return merge_locals(locals(), d)
