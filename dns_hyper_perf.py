"""
Nonlinear simulation of fluid-saturated hyperelastic solids.

Running simulation: sfepy-run dns_hyper_perf.py
"""
import os.path as osp
from functools import partial
import numpy as nm
import meshio
from sfepy.base.base import Struct, output
from sfepy.terms.terms_hyperelastic_ul import (NeoHookeanULTerm,
    HyperElasticULFamilyData)
from sfepy.discrete.functions import ConstantFunctionByRegion
from sfepy.terms.extmods.terms import sym2nonsym
import sfepy.linalg as la
from sfepy.solvers.ts import TimeStepper

wdir = osp.dirname(__file__)

hyperelastic_data = {
    'update_materials': True,
    'material_cache': {},
    'state': {'u': None, 'du': None,
              'p': None, 'dp': None},
    'mapping0': None,
    'coors0': None,
}

sym_eye = {
    2: nm.array([[1, 1, 0]]).T,
    3: nm.array([[1, 1, 1, 0, 0, 0]]).T,
}

nonsym_delta = {
    2: nm.array([[0, 0, 0, -1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [-1, 0, 0, 0]]),
    3: nm.array([[0, 0, 0, 0, -1, 0, 0, 0, -1],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [-1, 0, 0, 0, 0, 0, 0, 0, -1],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [-1, 0, 0, 0, -1, 0, 0, 0, 0]]),
}


def post_process(out, pb, state, extend=False):
    if isinstance(state, dict):
        pass
    else:
        he_state = hyperelastic_data['state']
        vars = pb.create_variables(['U', 'P'])
        vars['U'].set_data(he_state['u'])
        vars['P'].set_data(he_state['p'])

        ev = partial(pb.evaluate, mode='el_avg')
        evu = partial(ev, var_dict={'U': vars['U']})
        evp = partial(ev, var_dict={'P': vars['P']})

        if pb.conf.is_nonlinear == 'lin':
            stress = evu('ev_cauchy_stress.i.Omega(mat_he.As, U)')
            strain = evu('ev_cauchy_strain.i.Omega(U)')
            out['cauchy_strain'] = Struct(mode='cell', data=strain)
        else:
            stress = evu('ev_volume_integrate_mat.i.Omega(mat_he.S, U)')
            strain = evu('ev_volume_integrate_mat.i.Omega(mat_he.E, U)')
            out['green_strain'] = Struct(mode='cell', data=strain)

        out['cauchy_stress'] = Struct(mode='cell', data=stress)

        press = he_state['p'][:, nm.newaxis]
        out['p'] = Struct(mode='vertex', data=press)
        out['u'] = Struct(mode='vertex', data=he_state['u'])

        epress = evp('ev_integrate.i.Omega(P)')
        out['epress'] = Struct(mode='cell', data=epress)

        vol = evu('ev_volume.i.Omega(U)')
        eids = pb.conf.mmesh.cell_data['eid'][0]
        mat_id = pb.conf.mmesh.cell_data['mat_id'][0]
        astress = stress * vol

        nchs = pb.conf.n_channels
        aepress = [epress * vol for k in range(1 + nchs)]

        for eid in nm.unique(eids):
            idxs = eids == eid
            astress[idxs] = astress[idxs].sum(axis=0) / vol[idxs].sum(axis=0)

            for k in range(1 + nchs):
                idxsk = nm.logical_and(idxs, mat_id == (k + 1))
                volk = vol[idxsk].sum(axis=0)
                aepress[k][idxs] = aepress[k][idxsk].sum(axis=0) / volk

        out['acauchy_stress'] = Struct(mode='cell', data=astress)
        out['eid'] = Struct(mode='cell', data=eids[:, None, None, None])

        for k in range(1 + nchs):
            out[f'aepress_{k + 1}'] = Struct(mode='cell', data=aepress[k])

    return out


def get_mat(ts, coors, mode, term=None, problem=None, **kwargs):
    hyperela = hyperelastic_data
    ts = hyperela['ts']
    pb = problem

    output('get_mat: mode=%s, update=%s'\
        % (mode, hyperela['update_materials']))

    if not mode == 'qp':
        return

    geom_type = list(term.geometry_types.values())[0]
        
    material_cache = hyperela['material_cache']

    if not hyperela['update_materials']:
        return {k: nm.array(v) for k, v in material_cache.items()}

    if not hasattr(pb, 'family_data'):
        pb.family_data = HyperElasticULFamilyData()

    mvars = pb.create_variables(['U', 'P'])
    state_u, state_p = mvars['U'], mvars['P']

    if state_u.data[0] is None:
        state_u.init_data()

    state_u.set_data(pb.domain.get_mesh_coors(actual=True)
                     - pb.domain.get_mesh_coors(actual=False))
    state_u.field.clear_mappings()
    family_data = pb.family_data(state_u, term.region,
                                 term.integral, geom_type)

    if len(state_u.field.mappings0) == 0:
        state_u.field.save_mappings()

    npts = coors.shape[0]
    n_el, n_qp, dim, _, _ = state_u.get_data_shape(term.integral,
                                                   term.act_integration,
                                                   term.region.name)

    # relative displacement
    if hyperela['state']['du'] is not None:
        state_u.set_data(hyperela['state']['du'])
        grad_du_qp = state_u.evaluate(mode='grad',
            integral=term.integral).reshape((npts, dim, dim))
    else:
        grad_du_qp = nm.zeros((npts, dim, dim), dtype=nm.float64)

    div_du_qp = nm.trace(grad_du_qp, axis1=1, axis2=2).reshape((npts, 1, 1))
   
    if hyperela['state']['p'] is not None:
        state_p.set_data(hyperela['state']['p'])
        press_qp = state_p.evaluate(integral=term.integral).reshape((npts, 1, 1))
    else:
        press_qp = nm.zeros((npts, 1, 1), dtype=nm.float64)

    conf_mat = pb.conf.materials
    solid_key = [key for key in conf_mat.keys() if 'solid' in key][0]
    solid_mat = conf_mat[solid_key].values
    mat = {}
    for mat_key in ['mu', 'K']:
        if isinstance(solid_mat[mat_key], dict):
            mat_fun = ConstantFunctionByRegion({mat_key: solid_mat[mat_key]})
            mat0 = mat_fun.function(ts=ts, coors=coors, mode='qp',
                                    term=term, problem=pb)[mat_key]
            mat[mat_key] = mat0.reshape((n_el, n_qp) +  mat0.shape[-2:])
        else:
            mat[mat_key] = nm.ones((n_el, n_qp, 1, 1)) * solid_mat[mat_key]

    shape = family_data.green_strain.shape[:2]
    assert(npts == nm.prod(shape))
    sym = family_data.green_strain.shape[-2]
    dim2 = dim**2

    fargs = [family_data.get(name)
             for name in NeoHookeanULTerm.family_data_names]
    stress_eff = nm.empty(shape + (sym, 1), dtype=nm.float64)
    tanmod_eff = nm.empty(shape + (sym, sym), dtype=nm.float64)
    NeoHookeanULTerm.stress_function(stress_eff, mat['mu'], *fargs)
    NeoHookeanULTerm.tan_mod_function(tanmod_eff, mat['mu'], *fargs)

    stress_eff_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    tanmod_eff_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    sym2nonsym(stress_eff_ns, stress_eff)
    sym2nonsym(tanmod_eff_ns, tanmod_eff)

    J = family_data.det_f.reshape((npts, 1, 1))
    mtx_f = family_data.mtx_f.reshape((npts, dim, dim))

    stress_p = - press_qp * J * sym_eye[dim]

    mat_A = (tanmod_eff_ns + stress_eff_ns).reshape((npts, dim2, dim2))\
        + J * press_qp * nonsym_delta[dim]
    
    mtxI = nm.eye(dim)

    mat_B = (grad_du_qp - mtxI * div_du_qp).transpose(0, 2, 1)

    mat['K'] = mat['K'].reshape((npts, dim, dim))
    mat_H = div_du_qp * mat['K']\
        - la.dot_sequences(mat['K'], grad_du_qp, 'ABT')\
        - la.dot_sequences(grad_du_qp, mat['K'], 'ABT')

    if not pb.conf.is_nonlinear:
        mat_B *= 0
        mat_H *= 0
        mat_A = tanmod_eff_ns.reshape((npts, dim2, dim2))
        state_u.set_data(hyperela['state']['u'])
        grad_u_qp = state_u.evaluate(mode='grad', integral=term.integral)
        stress_eff_ns = la.dot_sequences(mat_A, grad_u_qp.reshape((-1, dim*dim, 1)))

        stress_eff = nm.empty((npts, sym, 1), dtype=nm.float64)
        stress_eff[:, 0] = stress_eff_ns[:, 0]
        stress_eff[:, 1] = stress_eff_ns[:, 3]
        stress_eff[:, 2] = 0.5 * (stress_eff_ns[:, 1] + stress_eff_ns[:, 2])

    out = {
        'E': 0.5 * (la.dot_sequences(mtx_f, mtx_f, 'ATB') - nm.eye(dim)),
        'S': (stress_eff.reshape((npts, sym, 1)) + stress_p) / J, # Cauchy stress
        'A': mat_A / J,
        'BI': mtxI + mat_B,
        'KH': mat['K'] + mat_H,
    }

    hyperela['material_cache'] = \
        {k: nm.array(v) for k, v in out.items()}

    return out


def incremental_algorithm(pb):
    hyperela = hyperelastic_data
    ts = pb.conf.tstep

    hyperela['ts'] = ts
    hyperela['ofn_trunk'] = pb.ofn_trunk + '_%03d'
    pb.domain.mesh.coors_act = pb.domain.mesh.coors.copy()

    pbvars = pb.get_variables()

    he_state = hyperela['state']

    out = []
    out_data ={}

    coors0 = pbvars['u'].field.get_coor()
    he_state['coors0'] = coors0.copy()
    he_state['u'] = nm.zeros_like(coors0)
    he_state['du'] = nm.zeros_like(coors0)

    press0 = pbvars['p'].field.get_coor()[:, 0].squeeze()
    he_state['p'] = nm.zeros_like(press0)
    he_state['dp'] = nm.zeros_like(press0)

    for step, time in ts:
        print('>>> step %d (%e):' % (step, time))
        hyperela['update_materials'] = True

        pb.ofn_trunk = hyperela['ofn_trunk'] % step

        pbvars['hP'].set_data(he_state['p'])

        yield pb, out
        
        state = out[-1][1]
        result = state.get_state_parts()
        du = result['u']

        he_state['u'] += du.reshape(he_state['du'].shape)
        he_state['du'][:] = du.reshape(he_state['du'].shape)
        if pb.conf.is_nonlinear == True:
            pb.set_mesh_coors(he_state['u'] + he_state['coors0'],
                            update_fields=True, actual=True, clear_all=False)

        dp = result['p']
        he_state['p'] += dp
        he_state['dp'][:] = dp

        out_data = post_process(out_data, pb, state, extend=False)
        filename = pb.get_output_name()
        pb.save_state(filename, out=out_data)

        yield None

        print('<<< step %d finished' % step)


def ramp_fun(nt, ramp=0.5):
    val = 0.5 + nm.sin(nm.pi/ramp * nt - nm.pi/2) / 2
    val = 1. if nt > ramp else val
    return val


def move_fun(ts, coor, ramp=0.5, incremental=True, **kwargs):
    pb = kwargs['problem']
    if hasattr(pb.conf, 'tstep'):
        ts = pb.conf.tstep

    nt_prev = ts.nt - ts.dt / ts.t1
    mul = ramp_fun(ts.nt, ramp)

    if nt_prev > 0 and incremental:
        mul = mul - ramp_fun(ts.nt - ts.dt / ts.t1, ramp)

    return coor * 0 + nm.array(pb.conf.displ_val) * mul


def define(filename_mesh='meshes/dns_32x16_perf.vtk', dim=2,
           n_channels=1,
           t_end=0.1, n_step=50,
           output_dir='output',
           displ_val=[0.03, 0],
           is_nonlinear=True,
           matK1=1e-6, matK3=1e-8):

    filename_mesh = osp.join(wdir, filename_mesh)
    mmesh = meshio.read(filename_mesh)
    x1 = mmesh.points[:, 0].max()

    tstep = TimeStepper(0, t_end, n_step=n_step)

    options = {
        'output_dir': output_dir,
        'parametric_hook': 'incremental_algorithm',
        'file_format': 'vtk',
    }

    chs = list(nm.arange(n_channels) + 1)
    mat_mu = {'Omega_m': 1e6}
    mat_K = {'Omega_m': matK3 * nm.eye(dim)}
    val_K = [matK1, matK1 * 10]
    regions = {}
    for ich, ch in enumerate(chs):
        mat_mu[f'Omega_{ch}'] = .6e6
        mat_K[f'Omega_{ch}'] = val_K[ich] * nm.eye(dim),
        regions[f'Omega_{ch}'] = f'cells of group {ich + 2}',

    materials = {
        'mat_he': 'get_mat',
        'solid': ({'mu': mat_mu, 'K': mat_K},),
    }

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement', 0),
        'v': ('test field', 'displacement', 'u'),
        'p': ('unknown field', 'pressure', 1),
        'q': ('test field', 'pressure', 'p'),
        'U': ('parameter field', 'displacement', '(set-to-None)'),
        'P': ('parameter field', 'pressure', '(set-to-None)'),
        'hP': ('parameter field', 'pressure', 'p'),
    }

    regions.update({
        'Omega': 'all',
        'Omega_m': 'cells of group 1',
        'Left': ('vertices in (x < 1e-6)', 'facet'),
        'Right': (f'vertices in (x > {x1 * (1 - 1e-6)})', 'facet'),
    })

    ebcs = {
        'l': ('Left', {'u.all': 0.0}),
        'r': ('Right', {'u.all': 'move'}),
    }

    functions = {
        'move': (move_fun,),
        'get_mat': (get_mat,),
    }

    integrals = {
        'i': 2,
    }

    equations = {
        'balance_of_forces': """
            dw_nonsym_elastic.i.Omega(mat_he.A, v, u)
          - dw_biot.i.Omega(mat_he.BI, v, p)
          =
          - dw_lin_prestress.i.Omega(mat_he.S, v)""",
        'mass_conservation': """
     - %e * dw_biot.i.Omega(mat_he.BI, u, q)
          - dw_diffusion.i.Omega(mat_he.KH, q, p) 
          =
            dw_diffusion.i.Omega(mat_he.KH, q, hP)""" % (1. / tstep.dt),
    }

    solvers = {    
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {
            'eps_a': 1e-8,
            'eps_r': 1e-3,
            'i_max': 1,
        }),
    }

    return locals()