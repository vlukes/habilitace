import os.path as osp
import numpy as nm
from scipy.io import savemat
import meshio
from sfepy.homogenization.utils import define_box_regions
from sfepy.homogenization.micmac import get_homog_coefs_linear
from sfepy.base.base import Struct
from sfepy.solvers.ts import TimeStepper
from sfepy.postprocess.probes_vtk import Probe, ProbeFromFile
from sfepy.homogenization.utils import iter_sym
from sfepy.linalg.utils import dot_sequences

wdir = osp.dirname(__file__)


def get_homog(pb, dargs=None):
    conf = pb.conf
    cache = pb.cache
    coefs_fname = osp.join(pb.output_dir, f'{conf.filename_coefs}.h5')

    if coefs_fname not in cache:
        def_args = {
            'eps0': conf.eps0,
            'filename_mesh': conf.filename_mesh_micro,
            'filename_coefs': conf.filename_coefs,
            'output_dir': conf.output_dir,
        }

        if hasattr(conf, 'micro_args'):
            def_args.update(conf.micro_args)

        if dargs is not None:
            def_args.update(dargs)

        print(f'coefs. file: {coefs_fname}')
        print(f'micro file: {conf.filename_micro}')

        coefs = get_homog_coefs_linear(0, 0, None,
                                       micro_filename=conf.filename_micro,
                                       coefs_filename=coefs_fname,
                                       define_args=def_args)

        cf_keys = coefs.keys()
        for k in cf_keys:
            v = coefs[k]
            if isinstance(v, float):
                coefs[k] = nm.array(v, dtype=nm.float64).reshape((1, 1, 1))
            elif isinstance(v, nm.ndarray):
                aux = nm.atleast_2d(v)
                if aux.shape[-1] > aux.shape[-2]:
                    aux = aux.T
                coefs[k] = aux.reshape((1,) + aux.shape)

        cache[coefs_fname] = coefs

    return cache[coefs_fname]


def eval_coefs_x(st_qp, coefs, exprs=None):
    if exprs is None:
        exprs = ['A|e|e', 'A|e|p',
                 'B|p|e', 'B|p|p',
                 'BT|e|e', 'BT|de|e', 'BT|e|p', 'BT|de|p',
                 'M|dp|e', 'M|dp|p',
                 'K|gp|e', 'K|gp|p']

    nqp = st_qp['e'].shape[0]
    dim = coefs['K'].shape[-1]

    coefs_x = {}
    for cf_str in exprs:
        cf, mul, op = cf_str.split('|')
        t_flag = False
        if cf[-1] == 'T':
            cf = cf[:-1]
            t_flag = True

        stval = st_qp[mul]

        if op == 'e':
            d1 = coefs[cf].shape[2] if t_flag else coefs[cf].shape[1]
            d2 = st_qp[op].shape[1]
            val = nm.zeros((nqp, d1, d2), dtype=nm.float64)
            for ii, irc in enumerate(iter_sym(dim)):
                key = 's%s_e%d%d' % ((cf,) + irc)
                cfval = coefs[key].transpose((0, 2, 1)) if t_flag\
                    else coefs[key]
                if mul[0] == 'e' or mul[:2] == 'gp' or mul[:2] == 'de':
                    val[..., ii] = dot_sequences(cfval, stval)[..., 0]
                elif mul[0] == 'p' or mul[0] == 'r'\
                        or mul[:2] == 'dp' or mul[:2] == 'dr':
                    val[..., ii] = (cfval * stval)[..., 0]
                else:
                    raise ValueError

        elif op[0] == 'p':
            key = f's{cf}_{op}'
            cfval = coefs[key].transpose((0, 2, 1)) if t_flag\
                else coefs[key]
            if mul[0] == 'e' or mul[:2] == 'gp' or mul[:2] == 'de':
                val = dot_sequences(cfval, stval)
            elif mul[0] == 'p' or mul[0] == 'r'\
                    or mul[:2] == 'dp' or mul[:2] == 'dr':
                val = cfval * stval
            else:
                raise ValueError

        else:
            raise ValueError

        if val.shape[-1] > val.shape[-2]:
            val = val.transpose((0, 2, 1))

        coefs_x[f's{cf}_{mul}{op}'] = val

    return coefs_x


def get_mat_eval_state(pb, state, int_reg, nqp, dim):
    sym = 3 * dim - 3
    out = {}
    evars = pb.create_variables(['U', 'P'])

    ir = '%s.%s' % int_reg
    ev = pb.evaluate
    
    evars['U'].set_data(state['u'])
    out['e'] = ev(f'ev_cauchy_strain.{ir}(U)', mode='qp', U=evars['U']).reshape((nqp, sym, 1))
    evars['U'].set_data(state['u_prev'])
    out['e0'] = ev(f'ev_cauchy_strain.{ir}(U)', mode='qp', U=evars['U']).reshape((nqp, sym, 1))
    out['de'] = out['e'] - out['e0']

    evars['P'].set_data(state['p'])
    out['p'] = ev(f'ev_volume_integrate.{ir}(P)', mode='qp', P=evars['P']).reshape((nqp, 1, 1))
    out['gp'] = ev(f'ev_grad.{ir}(P)', mode='qp', P=evars['P']).reshape((nqp, 3, 1))
    evars['P'].set_data(state['p_prev'])
    out['p0'] = ev(f'ev_volume_integrate.{ir}(P)', mode='qp', P=evars['P']).reshape((nqp, 1, 1))
    out['dp'] = out['p'] - out['p0']

    return out


def get_mat_nonlinear(pb, coefs, int_reg, nqp, dim, nlmul=1, nlmulK=1):
    st_qp = get_mat_eval_state(pb, pb.cache['state'], int_reg, nqp, dim)

    bar_eta = coefs['bar_eta']
    coefs_sa = pb.conf.required_coefs

    coefs_out = {}
    for cf in coefs_sa:
        cf_ = cf[:-3] if cf.endswith('eta') else cf
        val = coefs[f's{cf_}_p'] * st_qp['p']
        for ii, irc in enumerate(iter_sym(dim)):
            key = 's%s_e%d%d' % ((cf_,) + irc)
            val += coefs[key] * (st_qp['e'][:, ii, :].reshape(nqp, 1, 1))

        if cf == 'Keta':
            coefs_out[f't{cf}'] = (coefs[cf_] + val * nlmulK) / bar_eta
        else:
            coefs_out[f't{cf}'] = coefs[cf_] + val * nlmul

    coefs_x = eval_coefs_x(st_qp, coefs)

    coefs_out['bA'] = coefs_out['tA']\
        + (coefs_x['sA_ee'] - coefs_x['sB_pe']) * nlmul
    coefs_out['bB'] = coefs_out['tB']\
        + (coefs_x['sB_pp'] - coefs_x['sA_ep']) * nlmul
    coefs_out['bD'] = coefs_out['tB']\
        + (coefs_x['sB_dee'] + coefs_x['sM_dpe']) * nlmul
    coefs_out['bM'] = coefs_out['tM']\
        + (coefs_x['sM_dpp'] + coefs_x['sB_dep']) * nlmul
    coefs_out['bKeta'] = coefs_out['tKeta']
    coefs_out['bGeta'] = coefs_x['sK_gpe'].transpose((0, 2, 1))\
        / bar_eta * nlmulK
    coefs_out['bQeta'] = coefs_x['sK_gpp'] / bar_eta * nlmulK

    return coefs_out


def get_mat(ts, coors, mode=None, problem=None, **kwargs):
    if mode != 'qp':
        return

    print('>>> material function')

    cache = problem.cache

    term = kwargs['term']
    int_reg = term.integral.name, term.region.name
    cf_cache_name = 'mat_coefs_%s_%s' % int_reg

    if cf_cache_name not in cache:
        print('>>>   updating')
        coefs_ = get_homog(problem)
        req_coefs = problem.conf.required_coefs

        coefs = {k: v for k, v in coefs_.items()
                 if (k in req_coefs) and (k != 'Keta')}

        if 'Keta' in req_coefs:
            coefs['Keta'] = coefs_['K'] / coefs_['bar_eta']

        nqp, dim = coors.shape
        if problem.conf.nonlinear:
            coefs.update(get_mat_nonlinear(problem, coefs_, int_reg, nqp, dim))
        else:
            coefs['bA'] = coefs['tA'] = coefs['A']
            coefs['bB'] = coefs['tB'] = coefs['bD'] = coefs['B']
            coefs['bM'] = coefs['tM'] = coefs['M']
            coefs['bKeta'] = coefs['tKeta'] = coefs['Keta']
            coefs['bQeta'] = nm.zeros((1, dim, 1), dtype=nm.float64)
            coefs['bGeta'] = nm.zeros((1, dim, 3 * dim - 3), dtype=nm.float64)

        cache[cf_cache_name] = coefs

    return cache[cf_cache_name]


def write_output(pb, fname, out):
    fname = osp.join(pb.conf.options.get('output_dir', '.'), fname)

    mesh = pb.domain.mesh
    n_nod = mesh.n_nod
    n_el = mesh.n_el

    vars = pb.get_variables()

    out_ = {}
    for k, v in out.items():
        if hasattr(v, 'region') or hasattr(v, 'var_name'):
            if hasattr(v, 'region'):
                region = pb.domain.regions[v.region]
            elif hasattr(v, 'var_name'):
                region = vars[v.var_name].field.region

            nidxs = region.vertices
            eidxs = region.cells
            if v.mode == 'vertex':
                val = nm.zeros((n_nod,) + v.data.shape[1:], dtype=v.data.dtype)
                val[nidxs] = v.data
                v.data = val
            elif v.mode == 'cell':
                val = nm.zeros((n_el,) + v.data.shape[1:], dtype=v.data.dtype)
                val[eidxs] = v.data
                v.data = val
        else:
            out_[k] = v

    mesh.write(fname, out=out)


def eval_probes(pb, out, probe_lines=None, n_points=100, flag=''):
    bbox = pb.domain.get_mesh_bounding_box()
    p0 = bbox[0]
    dp = bbox[1] - bbox[0]

    if isinstance(out, str):
        probe = ProbeFromFile(out)
        out_vars = [k for k, _, _ in probe_lines]
    else:
        probe = Probe(out, pb.domain.mesh)
        out_vars = out.keys()

    pout = {
        f'time{flag}': pb.conf.tstep.time,
    }

    for var, field, line in probe_lines:
        if var not in out_vars:
            continue

        p1 = nm.array(line[0]) * dp + p0
        p2 = nm.array(line[1]) * dp + p0
        pname = f'line_{var}'
        probe.add_line_probe(pname, p1, p2, n_points)
        t, pval = probe(pname, var)

        if f'lc{flag}' not in pout:
            pout[f'lc{flag}'] = t[:, None]

        pout[f'{field}{flag}'] = pval

    return pout


def post_process(pb, state, region='Omega', mat='hom', integral='i2',
                 perm='Keta'):

    out_data = pb.cache['out_data']
    probe_flag = f'|{pb.conf.tstep.step}'
    pbvars = pb.get_variables()
    pbvars['U'].set_data(state['u'])
    pbvars['P'].set_data(state['p'])

    out = {}

    out['u'] = Struct(name='output_data', mode='vertex',
                      var_name='u', data=state['u'].reshape((-1, 3)))
    strain = pb.evaluate('ev_cauchy_strain.i2.Omega(U)', mode='el_avg')
    out['e'] = Struct(name='output_data', mode='cell',
                      var_name='u', region='Omega', data=strain)

    out['p'] = Struct(name='output_data', mode='vertex',
                      var_name='p', data=state['p'][:, None])
    dvel = pb.evaluate('ev_diffusion_velocity.i2.Omega(hom.Keta, P)', mode='el_avg')
    out['w'] = Struct(name='output_data', mode='cell',
                      var_name='p', region='Omega', data=dvel)

    fname = f'{pb.ofn_trunk}_{pb.conf.tstep.step:03d}.vtk'
    write_output(pb, fname, out)

    cline = [[0.05, 0.05, 0], [0.5, 0.5, 1]]
    probe_lines = [
        ('u', 'displacement', cline),
        ('e', 'strain', cline),
        ('p', 'pressure', cline),
        ('w', 'velocity', cline),
    ]
    
    if 'probes' not in out_data:
        out_data.update(eval_probes(pb, out, probe_lines, flag=probe_flag))

    return out


def post_process_final(pb):
    print('>>> final post-processing')
    cache = pb.cache

    out_data = cache['out_data']

    keys = [k for k in out_data.keys() if '|' not in k]
    out = {k: nm.array(out_data[k]) for k in keys}

    keys = {k.split('|')[0] for k in out_data.keys() if '|' in k}
    times = list({int(k.split('|')[1]) for k in out_data.keys() if '|' in k})

    keys1 = {k for k in keys if k.startswith('lc')}
    out.update({k: nm.array(out_data[f'{k}|0']).ravel() for k in keys1})
    keys2 = {k for k in keys if k.startswith('time')}
    out.update({k: nm.array([out_data[f'{k}|{t}'] for t in times]).ravel()
                for k in keys2})
    out.update({k: nm.stack([out_data[f'{k}|{t}'] for t in times])
                for k in (keys - keys1 - keys2)})

    fname = f'poroela_data_{'nl' if pb.conf.nonlinear else 'l'}.mat'
    fname = osp.join(pb.conf.options.get('output_dir', '.'), fname)

    savemat(fname, out)
    print(f'>>> data saved to {fname}')


def clear_cache(cache, key):
    keys = list(cache.keys())
    for k in keys:
        if key in k:
            del cache[k]


def time_stepping_iter(pb):
    pb.cache = cache = {'out_data': {}, 'iiter': 1}
    pbvars = pb.get_variables()
    tstep = pb.conf.tstep

    state = {
        'u': nm.zeros((pbvars['u'].n_dof,), dtype=nm.float64),
        'u_prev': nm.zeros((pbvars['u'].n_dof,), dtype=nm.float64),
        'p': nm.zeros((pbvars['p'].n_dof,), dtype=nm.float64),
        'p_prev': nm.zeros((pbvars['p'].n_dof,), dtype=nm.float64),
    }

    cache['state'] = state

    out = []
    for step, time in tstep:
        print('##################################################')
        print(f'  time: {time}, step: {step}')

        iiter = 1
        next_iteration = True

        while next_iteration:
            pbvars['U'].set_data(state['u'])
            pbvars['P'].set_data(state['p'])
            pbvars['U1'].set_data(state['u_prev'])
            pbvars['P1'].set_data(state['p_prev'])

            clear_cache(cache, 'mat_')
            cache['iiter'] = iiter
            
            yield pb, out

            new_state = out[-1][1].get_state_parts()
            state['u'] += new_state['u']
            state['p'] += new_state['p']

            if pb.conf.nonlinear:
                iter_norm = nm.linalg.norm(new_state['u'])
                
                if iiter == 1:
                    iter_norm0 = iter_norm.copy()

                if iter_norm0 > 0:
                    iter_norm /= iter_norm0

                print('--------------------------------------------------')
                print(f'  iter: {iiter}')
                print(f'  norm: {iter_norm}')

                if iter_norm < 1e-3: 
                    next_iteration = False
                    post_process(pb, state)
                elif iiter > 10:
                    print('maximal number of iterations!!!')
                    next_iteration = False
                    post_process(pb, state)

                iiter += 1
            else:
                next_iteration = False
                post_process(pb, state)

            yield None
    
        state['u_prev'][:] = state['u']
        state['p_prev'][:] = state['p']

        for k in ['u', 'p']:
            v = state[k]
            print(f'  {k:3}: min = {nm.min(v)}, max = {nm.max(v)}')
        print('##################################################')

    post_process_final(pb)


def ramp_fce(t, ramp=0.1):
    out = t / ramp

    if out > 1.0:
        out = 1.0

    return out


def move_bnd(ts, coors, problem=None, **kwargs):
    conf = problem.conf
    ts = conf.tstep

    uh = conf.u_bar
    val = ramp_fce(ts.time / ts.t1)

    if 'iiter' in problem.cache:
        # if ts.step > 0:
        #     val -= ramp_fce((ts.time - ts.dt) / ts.t1)
        if problem.cache['iiter'] == 1 and ts.step > 0:
            val -= ramp_fce((ts.time - ts.dt) / ts.t1)
        else:
            val = 0.

    print(f'>>> displ. function: {val}')

    return nm.ones((coors.shape[0], 1), dtype=nm.float64) * val * uh


def define(is_recovery=True,
           filename_mesh=osp.join(wdir, 'meshes', 'macro_poroela.vtk'),
           eps0=1e-4,
           u_bar=-0.005,
           t_end=2, t_nstep=50,
           output_dir='output',
           nonlinear=True,
           filename_mesh_micro=osp.join(wdir, 'meshes', 'micro_poroela_c.vtk'),
        #    filename_mesh_micro=osp.join(wdir, 'meshes', 'micro_poroela_o.vtk'),
           filename_coefs='coefs_poroela',
           filename_micro=osp.join(wdir, 'poroela_mikro.py')
           ):
   
    required_coefs = ['A', 'B', 'M', 'Keta']

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 1),
        'sfield': ('real', 'scalar', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement', 0, 1),
        'v': ('test field', 'displacement', 'u'),
        'U': ('parameter field', 'displacement', 'u'),
        'U1': ('parameter field', 'displacement', 'u'),
        'p': ('unknown field', 'pressure', 1, 1),
        'q': ('test field', 'pressure', 'p'),
        'P': ('parameter field', 'pressure', 'p'),
        'P1': ('parameter field', 'pressure', 'p'),
        'svar': ('parameter field', 'sfield',  'set-to-none'),
    }

    functions = {
        'get_mat': (get_mat,),
        'move_bnd': (move_bnd,),
    }

    materials = {
        'hom': 'get_mat',
    }

    integrals = {
        'i2': 2,
    }

    options = {
        'output_dir': output_dir,
        'parametric_hook': 'time_stepping_iter',
        # 'recovery_region_mode': 'tiled',  # 'el_centers'
        # 'recovery_eval_mode': 'continuous',  # 'constant'
    }

    aux_mesh = meshio.read(filename_mesh)
    region_box = [aux_mesh.points.min(axis=0), aux_mesh.points.max(axis=0)]

    regions = {
        'Omega': 'all',
        'Out': ('vertices of group 1', 'vertex'),
    }

    regions.update(define_box_regions(3, region_box[0], region_box[1]))

    ebcs = {
        'fixed_bottom': ('Bottom', {'u.2': 0.0}),
        'fixed_left': ('Left', {'u.0': 0.0}),
        'fixed_right': ('Right', {'u.0': 0.0}),
        'fixed_near': ('Near', {'u.1': 0.0}),
        'fixed_far': ('Far', {'u.1': 0.0}),
        'press_out': ('Out', {'p.0': 0.0}),
        'displ_in': ('Top', {'u.2': 'move_bnd'}),
    }

    solvers = {
        'ls': ('ls.mumps', {}),
        'nls': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-9,
            'eps_r': 1e-3,
            'problem': 'nonlinear',
        }),
    }

    tstep = TimeStepper(0.0, t_end, n_step=t_nstep)
    dt = tstep.dt

    equations = {
        'balance_of_forces': """
              dw_lin_elastic.i2.Omega(hom.bA, v, u)
            - dw_biot.i2.Omega(hom.bB, v, p)
            =
            - dw_lin_elastic.i2.Omega(hom.tA, v, U)
            + dw_biot.i2.Omega(hom.tB, v, P)
            """,
        'mass_conservation': f"""
            - dw_biot.i2.Omega(hom.bD, u, q)
            - dw_volume_dot.i2.Omega(hom.bM, q, p)
     - {dt} * dw_diffusion.i2.Omega(hom.bKeta, q, p)
     - {dt} * dw_diffusion_coupling.i2.Omega(hom.bQeta, q, p)
     - {dt} * dw_piezo_coupling.i2.Omega(hom.bGeta, u, q)
            =
            + dw_biot.i2.Omega(hom.tB, U, q)
            - dw_biot.i2.Omega(hom.tB, U1, q)
            + dw_volume_dot.i2.Omega(hom.tM, q, P)
            - dw_volume_dot.i2.Omega(hom.tM, q, P1)
     + {dt} * dw_diffusion.i2.Omega(hom.tKeta, q, P)""",
    }

    return locals()
