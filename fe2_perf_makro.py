"""
Two-scale nonlinear simulation of fluid-saturated hyperelastic solids.

Running simulation: sfepy-run fe2_perf_makro.py
"""
import os.path as osp
from functools import partial
import numpy as nm
from sfepy.base.base import Struct, output
from sfepy.discrete.fem import Mesh
from sfepy.homogenization.micmac import get_homog_coefs_nonlinear
from sfepy.linalg.utils import invs_fast, dot_sequences
from sfepy.solvers.ts import TimeStepper
from dns_hyper_perf import ramp_fun

wdir = osp.dirname(__file__)

hyperelastic_data = {
    'micro': {
        'mtx_f': None,
        'coefs': {},
    },
    'macro': {
        'problem': None,
        'pressures': [],
    },
}


def post_process_final(pb, state):
    from scipy.io import savemat

    if pb.conf.tstep.nt == 1.:
        data = hyperelastic_data['micro']['coefs']
        keys = data[0].keys()
        n_step = pb.conf.tstep.n_step
        out = {k: nm.array([data[t][k] for t in range(n_step)]) for k in keys}
        out['times'] = pb.conf.tstep.times

        data = hyperelastic_data['macro']['pressures']
        out['pressure'] = nm.array(data)

        fname = osp.join(pb.conf.options['output_dir'],
                         'cf_qp_nonlinear_perf.mat')
        savemat(fname, out)


def post_process(out, pb, state, extend=False):
    state = pb.cache['state']
    svars = pb.get_variables()
    svars.set_state_parts({'u': state['u']})


    ev = partial(pb.evaluate, mode='el_avg')
    oStruct = partial(Struct, name='output_data')

    stress = ev('ev_integrate_mat.i.Omega(solid.S, u)')
    strain = ev('ev_integrate_mat.i.Omega(solid.E, u)')
    ret_stress = ev('ev_integrate_mat.i.Omega(solid.Q, u)')

    out['cauchy_stress'] = oStruct(mode='cell', data=stress)
    out['retardation_stress'] = oStruct(mode='cell', data=ret_stress)
    out['green_strain'] = oStruct(mode='cell', data=strain)

    pqp = []
    for ch in pb.conf.chs:
        plab = f'p{ch}'
        svars.set_state_parts({plab: state[plab]})
        out[plab] = oStruct(mode='vertex', data=state[plab][:, None])
        dvel = ev(f'ev_diffusion_velocity.i.Omega(solid.C{ch}, p{ch})')
        out[f'w{ch}'] = oStruct(mode='cell', data=dvel)
        pqp.append(pb.evaluate('ev_integrate.i.Omega(p1)', mode='qp'))

    hyperelastic_data['macro']['pressures'].append(pqp)

    dim = pb.domain.mesh.dim
    out['u'] = oStruct(mode='vertex', data=state['u'].reshape((-1, dim)))

    return out


def get_rel_defgraf(mtx_f0, mtx_f):
    mtx_f0_inv = invs_fast(mtx_f0[None, ...])[0, ...]
    return dot_sequences(mtx_f, mtx_f0_inv, 'AB')


def get_homog_mat(ts, coors, mode, term=None, problem=None, **kwargs):
    if not mode == 'qp':
        return

    micro_data = hyperelastic_data['micro']
    ccache = micro_data['coefs']
    step = problem.conf.tstep.step

    iname, rname = term.integral.name, term.region.name
    output(f'>>> mat. fun: step={step}, integral={iname}, region={rname}')
    ckey = step

    if ckey not in ccache:
        ev = partial(problem.evaluate, mode='qp')
        
        state = problem.cache['state']
        
        dim = problem.domain.mesh.dim
        nqp = coors.shape[0]

        svars = problem.get_variables()
        state_u = svars['u']
        if len(state_u.field.mappings0) == 0:
            state_u.field.get_mapping(term.region, term.integral,
                                      term.integration)
            state_u.field.save_mappings()

        state_u.field.clear_mappings()
        svars.set_state_parts({'u': state['u']})

        mtx_f = ev('ev_def_grad.i.Omega(u)').reshape(-1, dim, dim)

        mtx_f_prev = micro_data['mtx_f']
        # relative deformation gradient
        if mtx_f_prev is not None:
            mtx_f_rel = get_rel_defgraf(mtx_f_prev, mtx_f)
        else:
            mtx_f_rel = mtx_f

        micro_data['mtx_f'] = mtx_f.copy()

        # relative macro strain
        macro_data = {'mtx_e_rel': mtx_f_rel - nm.eye(dim)}

        for ch in problem.conf.chs:
            plab = f'p{ch}'
            aux = f'i.Omega({plab})'
            svars.set_state_parts({plab: state[plab]})
            macro_data[plab] = ev(f'ev_integrate.{aux}').reshape(-1, 1, 1)
            macro_data['g' + plab] = ev(f'ev_grad.{aux}').reshape(-1, dim, 1)

            svars.set_state_parts({plab: state['d' + plab]})
            macro_data['d' + plab] = ev(f'ev_integrate.{aux}').reshape(-1, 1, 1)
            macro_data['gd' + plab] = ev(f'ev_grad.{aux}').reshape(-1, dim, 1)

        nel = term.region.entities[-1].shape[0]
        nqpe = nqp / nel
        if len(problem.conf.recovery_idxs):
            ridxs = nm.array(problem.conf.recovery_idxs)
            macro_data['recovery_idxs'] = ridxs[:, 0] * nqpe + ridxs[:, 1]
        else:
            macro_data['recovery_idxs'] = []

        ccoors_, macro_data_ = coors, macro_data #!!!!

        macro_data_['macro_ccoor'] = ccoors_
        macro_data_['step'] = step

        out = get_homog_coefs_nonlinear(ts, ccoors_, mode, macro_data_,
                                        term=term, problem=problem,
                                        iteration=ts.step, **kwargs)

        # Green strain
        out['E'] = 0.5 * (dot_sequences(mtx_f, mtx_f, 'ATB') - nm.eye(dim))

        for ch in problem.conf.chs:
            out['B%d' % ch] = out['B%d' % ch].reshape((nqp, dim, dim)) 
        out['Q'] = out['Q'].reshape((nqp, dim, dim))

        out['mtx_f'] = mtx_f
        ccache[ckey] = {k: nm.ascontiguousarray(v) for k, v in out.items()}

    else:
        output('>>>  cached')

    return ccache[ckey]


def time_stepping_fun(pb):
    pb.cache = cache = {}
    pbvars = pb.get_variables()
    tstep = pb.conf.tstep

    ofn_trunk = pb.ofn_trunk[:]

    pb.domain.mesh.coors_act = pb.domain.mesh.coors.copy()

    state_vars = pb.conf.state_vars
    state = {k: nm.zeros((pbvars[k].n_dof,), dtype=nm.float64)
             for k in state_vars}
    state.update({f'd{k}': v.copy() for k, v in state.items()})
    cache['state'] = state

    coors0 = pbvars['u'].field.get_coor().copy()

    out = []
    for step, time in tstep:
        output('##################################################')
        output(f'  time: {time}, step: {step}')

        pb.ofn_trunk = f'{ofn_trunk}_{step:03d}'
        
        new_coors = coors0 + state['u'].reshape(coors0.shape)
        pb.set_mesh_coors(new_coors,
                          update_fields=True, actual=True, clear_all=False)

        for ch in pb.conf.chs:
            pbvars[f'P{ch}'].set_data(state[f'p{ch}'])

        yield pb, out

        new_state = out[-1][1].get_state_parts()

        for k in state_vars:
            state[k] += new_state[k]
            state[f'd{k}'][:] = new_state[k] 

        yield None

        output('<<< step %d finished' % step)


def load_fun(ts, coor, mode=None, problem=None, ramp=0.5, **kwargs):
    if mode == 'qp':
        ts = problem.conf.tstep
        fun = ramp_fun(ts.nt, ramp)
        mul = nm.array(problem.conf.force_val) * fun
        val = nm.zeros((coor.shape[0], 2, 1), dtype=nm.float64)
        val[..., 0] = coor[:, [1]] / 0.2 * mul
        output(f'>>> load: {mul}')

        return {'val': val}


def define(filename_mesh='meshes/macro_L2.vtk',
           eps0=1e-3, recovery_idxs=[],
           filename_mesh_micro='meshes/micro_perf_2ch.vtk',
           n_channels=2,
           output_dir='output',
           t_end=0.1, t_nstep=50,
           force_val=[1e5, 0],
           multi=True,
           integration='reduced',  # reduced|full
           ):
    
    filename_mesh = osp.join(wdir, filename_mesh)
    filename_mesh_micro = osp.join(wdir, filename_mesh_micro)

    mesh = Mesh.from_file(filename_mesh)
    x1 = mesh.coors[:, 0].max()

    chs = nm.arange(n_channels) + 1
    
    tstep = TimeStepper(0.0, t_end, n_step=t_nstep)
    idt = 1. / tstep.dt

    options = {
        'output_dir': output_dir,
        'parametric_hook': 'time_stepping_fun',
        'micro_filename': osp.join(wdir, 'fe2_perf_mikro.py'),
        'post_process_hook': post_process,
        'post_process_hook_final': post_process_final,
    }

    functions = {
        'load_fun': (load_fun,),
        'get_homog': (lambda ts, coors, mode, **kwargs:
                      get_homog_mat(ts, coors, mode,
                                    define_args=micro_args, **kwargs),),
    }

    materials = {
        'solid': 'get_homog',
        'load' : (None, 'load_fun'),
    }

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement', 0),
        'v': ('test field', 'displacement', 'u'),
    }

    state_vars = ['u']
    for k in range(n_channels):
        variables[f'p{k + 1}'] = ('unknown field', 'pressure', 1)
        variables[f'q{k + 1}'] = ('test field', 'pressure', f'p{k + 1}')
        variables[f'P{k + 1}'] = ('parameter field', 'pressure', f'p{k + 1}')
        state_vars.append(f'p{k + 1}')

    regions = {
        'Omega': 'all',
        'Left': ('vertices in (x < 1e-6)', 'facet'),
        'Right': ('vertices in (x > %e)' % (x1 * (1 - 1e-6)), 'facet'),
    }

    ebcs = {
        'fix_left': ('Left', {'u.all': 0.0}),
    }

    micro_args = {
        'eps0': eps0,
        'dt': tstep.dt,
        'nch': n_channels,
        'dim': mesh.dim,
        'filename_mesh': filename_mesh_micro,
        'multiprocessing': multi,
        'output_dir': output_dir,
    }

    if integration == 'reduced':
        # 3 point quadrature rule
        integrals = {'i': 1}
    else:
        # 4 point quadrature rule
        integrals = {'i': {'name': 'i', 'order': 3, 'full_order': True}}

    equations = {
        'balance_of_forces': """
              dw_nonsym_elastic.i.Omega(solid.A, v, u)
            - dw_biot.i.Omega(solid.B1, v, p1)
            - dw_biot.i.Omega(solid.B2, v, p2)
            =
              dw_surface_ltr.i.Right(load.val, v)
            - dw_lin_prestress.i.Omega(solid.S, v)
            - dw_lin_prestress.i.Omega(solid.Q, v)
            """,
        'mass_conservation_1': f"""
    + {idt} * dw_biot.i.Omega(solid.B1, u, q1)
            + dw_dot.i.Omega(solid.G11, q1, p1)
            + dw_dot.i.Omega(solid.G12, q1, p2)
            + dw_diffusion.i.Omega(solid.C1, q1, p1)
            =
            - dw_volume_lvf.i.Omega(solid.Z1, q1)
            - dw_diffusion.i.Omega(solid.C1, q1, P1)
            """,
        'mass_conservation_2': f"""
    + {idt} * dw_biot.i.Omega(solid.B2, u, q2)
            + dw_dot.i.Omega(solid.G21, q2, p1)
            + dw_dot.i.Omega(solid.G22, q2, p2)
            + dw_diffusion.i.Omega(solid.C2, q2, p2)
            =
            - dw_volume_lvf.i.Omega(solid.Z2, q2)
            - dw_diffusion.i.Omega(solid.C2, q2, P2)
            """,
    }

    solvers = {
        'ls': ('ls.pypardiso', {}),
        'newton': ('nls.newton', {
            'eps_a': 1e-4,
            'eps_r': 1e-3,
            'i_max': 1,
        }),
    }

    return locals()
