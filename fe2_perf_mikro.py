"""
Two-scale nonlinear simulation of fluid-saturated hyperelastic solids.

Running simulation: sfepy-run fe2_perf_makro.py
"""
import numpy as nm
import os.path as osp
from sfepy.homogenization.utils import define_box_regions
import sfepy.homogenization.coefs_base as cb
import sfepy.discrete.fem.periodic as per
from sfepy.base.base import Struct, output, get_default
from sfepy.terms.terms_hyperelastic_ul import HyperElasticULFamilyData, NeoHookeanULTerm
from sfepy.terms.extmods.terms import sym2nonsym
from sfepy.discrete.functions import ConstantFunctionByRegion
import sfepy.linalg as la
from sfepy.base import multiproc

wdir = osp.dirname(__file__)
mp_module, _ = multiproc.get_multiproc()
multiproc_dependencies = mp_module.get_dict('dependencies', clear=True)
material_cache = mp_module.get_dict('material_cache', clear=True)

sym_eye = {
    2: nm.array([[1, 1, 0]]).T,
    3: nm.array([[1, 1, 1, 0, 0, 0]]).T,
}

nonsym_eye = {
    2: 'nm.array([1., 0., 0., 1.])',
    3: 'nm.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])',
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


class CorrStatePressureCh(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        micro_state, im = problem.micro_state
        macro_data = problem.homogenization_macro_data

        pvar = self.variable

        if micro_state[pvar] is not None:
            coors = problem.fields['pressure' + pvar[-1]].coors
            press = nm.dot(coors, macro_data['g' + pvar][im])[:, 0] \
                + micro_state[pvar][im] - macro_data[pvar][im][:, 0]
        else:
            ndof = problem.fields['pressure' + pvar[-1]].n_vertex_dof
            press = nm.zeros((ndof,), dtype=nm.float64)

        corr_sol = cb.CorrSolution(name=self.name,
                                   state={pvar: press})

        return corr_sol


class CorrStatePressureM(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        micro_state, im = problem.micro_state

        if micro_state['p'] is not None:
            press = micro_state['p'][im]
        else:
            ndof = problem.fields['pressure'].n_vertex_dof
            press = nm.zeros((ndof,), dtype=nm.float64)

        corr_sol = cb.CorrSolution(name=self.name,
                                   state={'p': press})

        return corr_sol


def post_process_hook(pb, nd_data, qp_data, ccoor, im, tstep, eps0,
                      recovery_file_tag=''):
    from sfepy.discrete.fem import Mesh

    elavg_data = {}
    # vol = qp_data['vol']
    # elvol = nm.sum(vol, axis=1)
    for k in qp_data.keys():
        elavg_data[k] = nm.average(qp_data[k], axis=1)[:, None, ...]
        # elavg_data[k] = (nm.sum(qp_data[k] * vol, axis=1) / elvol)[:, None, ...]

    output_dir = pb.conf.options.get('output_dir', '.')
    suffix = '%03d.%03d' % (im, tstep)
    coors = pb.get_mesh_coors(actual=True)
    coors = (coors - 0.5*(nm.max(coors, axis=0)
             - nm.min(coors, axis=0))) * eps0 + ccoor

    # Y
    out = {}
    out['displacement'] = Struct(name='output_data', mode='vertex',
                                 data=nd_data['u'] * eps0, variable='u')
    out['green_strain'] = Struct(name='output_data', mode='cell',
                                 data=elavg_data['E'])

    mesh = Mesh.from_region(pb.domain.regions['Y'], pb.domain.mesh)
    mesh.coors[:] = coors

    micro_name = pb.get_output_name(extra='recovered_Y_'
                                    + recovery_file_tag + suffix)
    filename = osp.join(output_dir, osp.basename(micro_name))

    output('  %s' % filename)
    mesh.write(filename, io='auto', out=out)

    p_tab = {'Ym': 'p'}
    p_tab.update({f'Yc{ch}': f'p{ch}' for ch in pb.conf.chs})
    mesh0 = pb.domain.mesh
    for rname in ['Ym'] + ['Yc%d' % ch for ch in pb.conf.chs]:
        reg = pb.domain.regions[rname]
        cells = reg.get_cells()

        out = {}
        out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                      data=elavg_data['S'][cells])
        out['velocity'] = Struct(name='output_data', mode='cell',
                                 data=elavg_data['w'][cells])
        out['pressure'] = Struct(name='output_data', mode='vertex',
                                 data=nd_data[p_tab[rname]][:, None])

        ac = nm.ascontiguousarray
        conn = mesh0.cmesh.get_cell_conn()
        cells = reg.entities[-1]
        verts = reg.entities[0]
        aux = nm.diff(conn.offsets)
        assert nm.sum(nm.diff(aux)) == 0
        conn = ac(conn.indices.reshape((mesh0.n_el, aux[0]))[cells])
        remap = -nm.ones(mesh0.n_nod)
        remap[verts] = nm.arange(verts.shape[0])
        conn = remap[conn]

        mesh = Mesh.from_data('region_%s' % rname,
                              ac(coors[verts]),
                              ac(mesh0.cmesh.vertex_groups[verts]),
                              [conn],
                              [ac(mesh0.cmesh.cell_groups[cells])],
                              [mesh0.descs[0]])

        micro_name = pb.get_output_name(extra='recovered_%s_' % rname 
                                        + recovery_file_tag + suffix)
        filename = osp.join(output_dir, osp.basename(micro_name))

        output('  %s' % filename)
        mesh.write(filename, io='auto', out=out)


def save_state(mesh, ndval, qpval, flag=''):
    out = {}
    for k, v in ndval.items():
        out[k] = Struct(mode='vertex', data=v)

    for k, v in qpval.items():
        ve = nm.average(v, axis=1)[:, None, ...]
        out[k] = Struct(mode='cell', data=ve)

    mesh.write(f'0state{flag}.vtk', out=out)


def get_hyperelastic_Y(pb, term, micro_state, im, region_name='Y'):
    from sfepy.terms import Term

    geom_type = list(term.geometry_types.values())[0]
    integral = term.integral
    act_integration = term.act_integration

    region = pb.domain.regions[region_name]
    el = region.get_cells().shape[0]
    nqp = tuple(term.integral.qps.values())[0].n_point
    npts = el * nqp

    mvars = pb.create_variables(
        ['U', 'P'] + ['P%d' % ch for ch in pb.conf.chs])
    state_u, state_p = mvars['U'], mvars['P']

    termY = Term.new('ev_grad(U)', term.integral, region, U=mvars['U'])
    termY.act_integration = act_integration

    if state_u.data[0] is None:
        state_u.init_data()

    u_mic = micro_state['coors'][im] - pb.domain.get_mesh_coors(actual=False)
    state_u.set_data(u_mic)
    state_u.field.clear_mappings()
    family_data = pb.family_data(state_u, region, integral, geom_type)

    if len(state_u.field.mappings0) == 0:
        state_u.field.save_mappings()

    n_el, n_qp, dim, _, _ = state_u.get_data_shape(integral, act_integration,
                                                   region_name)

    # relative displacement
    state_u.set_data(micro_state['coors'][im] - micro_state['coors_prev'][im]) # \bar u (du_prev)
    grad_du_qp = state_u.evaluate(mode='grad',
                                  integral=integral).reshape((npts, dim, dim))
    div_du_qp = nm.trace(grad_du_qp, axis1=1, axis2=2).reshape((npts, 1, 1))

    press_qp = nm.zeros((n_el, n_qp, 1, 1), dtype=nm.float64)
    grad_press_qp = nm.zeros((n_el, n_qp, dim, 1), dtype=nm.float64)
    press_nd = nm.zeros((u_mic.shape[0], 1), dtype=nm.float64)

    if micro_state['p'] is not None:
        p_mic = micro_state['p'][im]
        state_p.set_data(p_mic)
        cells = state_p.field.region.get_cells()
        press_qp[cells, ...] = state_p.evaluate(integral=term.integral)
        grad_press_qp[cells, ...] = state_p.evaluate(mode='grad',
                                                     integral=term.integral)

        press_nd[state_p.field.region.vertices, 0] = p_mic

        pch_mic = {}
        for ch in pb.conf.chs:
            state_pi = mvars['P%d' % ch]
            pch_mic[ch] = micro_state['p%d' % ch][im]
            state_pi.set_data(micro_state['p%d' % ch][im])
            cells = mvars['P%d' % ch].field.region.get_cells()
            press_qp[cells, ...] = state_pi.evaluate(integral=term.integral)
            grad_press_qp[cells, ...] = state_pi.evaluate(mode='grad',
                                                          integral=term.integral)

            press_nd[state_pi.field.region.vertices, 0] = pch_mic[ch]
    else:
        p_mic = nm.zeros((state_p.n_dof,), dtype=nm.float64)
        pch_mic = {ch: nm.zeros((mvars['P%d' % ch].n_dof,), dtype=nm.float64)
                   for ch in pb.conf.chs}

    if pb.conf.debug:
        ndval = {'u': u_mic.copy(), 'p': press_nd}
        qpval = {'p3e': press_qp.copy(), 'gp3e': grad_press_qp.copy()}

        save_state(pb.domain.mesh, ndval, qpval) 

    press_qp = press_qp.reshape((npts, 1, 1))
    grad_press_qp = grad_press_qp.reshape((npts, dim, 1))

    conf_mat = pb.conf.materials
    solid_key = [key for key in conf_mat.keys() if 'solid' in key][0]
    solid_mat = conf_mat[solid_key].values
    mat = {}
    for mat_key in ['mu', 'K']:
        if isinstance(solid_mat[mat_key], dict):
            mat_fun = ConstantFunctionByRegion({mat_key: solid_mat[mat_key]})
            mat0 = mat_fun.function(ts=None, coors=nm.empty(npts), mode='qp',
                                    term=termY, problem=pb)[mat_key]
            mat[mat_key] = mat0.reshape((n_el, n_qp) + mat0.shape[-2:])
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
    mat_BI = (mtxI * div_du_qp - grad_du_qp).transpose(0, 2, 1) + mtxI

    mat['K'] = mat['K'].reshape((npts, dim, dim))
    mat_H = div_du_qp * mat['K']\
        - la.dot_sequences(mat['K'], grad_du_qp, 'ABT')\
        - la.dot_sequences(grad_du_qp, mat['K'], 'ABT')

    out = {
        'E': 0.5 * (la.dot_sequences(mtx_f, mtx_f, 'ATB') - nm.eye(dim)),  # Green strain
        'S': (stress_eff.reshape((npts, sym, 1)) + stress_p) / J,  # Cauchy stress
        'A': mat_A / J,  # tangent elastic tensor, eq. (20)
        'BI': mat_BI,
        'KH': mat['K'] + mat_H,
        'H': mat_H,
        'w': -la.dot_sequences(mat['K'], grad_press_qp),  # perfusion velocity
    }

    return out


def def_mat(ts, coors, mode=None, term=None, problem=None, **kwargs):
    if not (mode == 'qp'):
        return

    pb = problem

    if not hasattr(pb, 'family_data'):
        pb.family_data = HyperElasticULFamilyData()

    macro_data = pb.homogenization_macro_data
    micro_state, im = pb.micro_state
    mac_id = micro_state['id'][im]
    step = macro_data['step']

    ckey = ('Y', term.integral.name, term.integration, mac_id, step)

    cached = ckey in material_cache
    output(f'>>> micro mat fun: mac_id={mac_id}, step={step}, cached={cached}, ckey={ckey}')
    if not cached:
        out = get_hyperelastic_Y(pb, term, micro_state, im)
        material_cache[ckey] = out

        if ('recovery_idxs' in macro_data
                and mac_id in macro_data['recovery_idxs']
                and step > 0):
            output(f'>>> recovery: mac_id={mac_id}, step={step}')

            nel = len(term.region.cells)
            nqp = coors.shape[0] // nel
            assert(nel * nqp == coors.shape[0])
            qp_data = {k: out[k].reshape((nel, nqp) + out[k].shape[1:])
                       for k in ['S', 'E', 'w']}

            pvals = ['p'] + [f'p{ch}' for ch in pb.conf.chs]
            nodal_data = {k: micro_state[k][im] for k in pvals}
            nodal_data['u'] =\
                micro_state['coors'][im] - pb.domain.get_mesh_coors(actual=False)

            post_process_hook(pb, nodal_data, qp_data,
                              macro_data['macro_ccoor'][im],
                              mac_id, macro_data['step'], pb.conf.eps0)
    else:
        out = material_cache[ckey]

    npts = coors.shape[0]
    region = term.region

    if region.name == 'Y':
        return out
    else:
        el = region.get_cells()[:, nm.newaxis]
        nel = el.shape[0]
        nqp = tuple(term.integral.qps.values())[0].n_point

        assert(npts == nel * nqp)

        idxs = (el * nm.array([nqp] * nqp) + nm.arange(nqp)).flatten()
        lout = {k: v[idxs, ...] for k, v in out.items()}

        return lout


def define(eps0=None, dt=None, nch=None, dim=None, filename_mesh=None,
           multiprocessing=None, output_dir=None,
           approx_u=1, approx_p=1, debug=True):
           
    chs = list(nm.arange(nch) + 1)
    update_u_by_p = [('corrs_%d' % ch, 'u', 'dp%d' % ch) for ch in chs]
    update_p_by_p = [('corrs_%d' % ch, 'p', 'dp%d' % ch) for ch in chs]

    micro_update = {
        'coors_prev': None,
        'coors': [('corrs_rs', 'u', 'mtx_e_rel'),
                  ('corrs_p', 'u', None)] + update_u_by_p,
        'p': [('corrs_rs', 'p', 'mtx_e_rel'),
              ('corrs_p', 'p', None)] + update_p_by_p,
    }
    for ch in chs:
        # micro_update[f'p{ch}'] = [(f'corrs_eta{ch}', f'p{ch}', f'gdp{ch}', eps0),
        #                           (f'corrs_p{ch}', f'p{ch}', None, eps0),
        #                           (None, None, f'dp{ch}')]
        micro_update[f'p{ch}'] = [(f'corrs_eta{ch}', f'p{ch}', f'gdp{ch}', eps0),
                                #   (f'corrs_p{ch}', f'p{ch}', None, 0),
                                  (None, None, f'dp{ch}')]


    options = {
        'coefs': 'coefs',
        'requirements': 'requirements',
        'volume': {'expression': 'ev_volume.5.Y(u)'},
        'output_dir': output_dir,
        'coefs_filename': 'coefs_hp',
        'multiprocessing': multiprocessing,
        'file_per_var': True,
        'micro_update': micro_update,
        'mesh_update_variable': 'u',
        'file_format': 'vtk',
    }

    fields = {
        'displacement': ('real', 'vector', 'Y', approx_u),
        'pressure': ('real', 'scalar', 'Ym', approx_p),
    }

    functions = {
        'match_x_plane': (per.match_x_plane,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
        'mat_fce': (def_mat,),
    }

    integrals = {
        'i': 3,
    }

    mat_mu = {'Ym': 1e6}
    mat_K = {'Ym': 1e-11 * nm.eye(dim) / eps0**2}
    val_K = [1e-6, 1e-5]
    val_mu = [.6e6, .8e6]
    for ich, ch in enumerate(chs):
        mat_mu[f'Yc{ch}'] = val_mu[ich]
        mat_K[f'Yc{ch}'] = val_K[ich] * nm.eye(dim),

    materials = {
        'mat_he': 'mat_fce',
        'solid': ({'mu': mat_mu, 'K': mat_K},),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'Piu': ('parameter field', 'displacement', 'u'),
        'Pi1u': ('parameter field', 'displacement', '(set-to-None)'),
        'Pi2u': ('parameter field', 'displacement', '(set-to-None)'),
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'Pip': ('parameter field', 'pressure', 'p'),
        'Pi1p': ('parameter field', 'pressure', '(set-to-None)'),
        'Pi2p': ('parameter field', 'pressure', '(set-to-None)'),
        'U': ('parameter field', 'displacement', '(set-to-None)'),
        'P': ('parameter field', 'pressure', '(set-to-None)'),
    }

    regions = {
        'Y': 'all',
        'Ym': 'cells of group 1',
    }

    if nch > 1:
        regions.update({
            'Gamma_mc': (' +s '.join(['r.Gamma%d' % ii for ii in chs]),
                         'facet', 'Ym')
        })
    else:
        regions.update({'Gamma_mc': ('copy r.Gamma1', 'facet')})

    regions.update(define_box_regions(dim, (0, 0, 0)[:dim], (1, 1, 1)[:dim]))

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
        'fixed_p': ('Gamma_mc', {'p.0': 0.0}),
    }

    epbcs = {
        'periodic_ux': (['Right', 'Left'], {'u.all': 'u.all'}, 'match_x_plane'),
        'periodic_px': (['Right', 'Left'], {'p.0': 'p.0'}, 'match_x_plane'),
    }

    periodic_all = ['periodic_ux', 'periodic_uy', 'periodic_px', 'periodic_py']

    if dim == 3:
        epbcs.update({
            'periodic_uy': (['Near', 'Far'], {'u.all': 'u.all'}, 'match_y_plane'),
            'periodic_uz': (['Bottom', 'Top'], {'u.all': 'u.all'}, 'match_z_plane'),
            'periodic_py': (['Near', 'Far'], {'p.0': 'p.0'}, 'match_y_plane'),
            'periodic_pz': (['Bottom', 'Top'], {'p.0': 'p.0'}, 'match_z_plane'),        })
        periodic_all += ['periodic_uz', 'periodic_pz']
    else:
        epbcs.update({
            'periodic_uy': (['Top', 'Bottom'], {'u.all': 'u.all'}, 'match_y_plane'),
            'periodic_py': (['Top', 'Bottom'], {'p.0': 'p.0'}, 'match_y_plane'),
        })

    lcbcs = {}

    coefs = {
        'A1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs'],
            'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, Pi1u, Pi2u)',
            'set_variables': [('Pi1u', ('pis_u', 'corrs_rs'), 'u'),
                              ('Pi2u', ('pis_u', 'corrs_rs'), 'u')],
            'class': cb.CoefNonSymNonSym,
        },
        'A2': {
            'status': 'auxiliary',
            'requires': ['corrs_rs'],
            'expression': 'dw_diffusion.i.Ym(mat_he.KH, Pi1p, Pi2p)',
            'set_variables': [('Pi1p', 'corrs_rs', 'p'),
                              ('Pi2p', 'corrs_rs', 'p')],
            'class': cb.CoefNonSymNonSym,
        },
        'A': {  # effective viscoelastic incremental tensor, eq. (51)
            'requires': ['c.A1', 'c.A2'],
            'expression': 'c.A1 + %e * c.A2' % dt,
            'class': cb.CoefEval,
        },
        'S': {  # averaged Cauchy stress, eq. (53)
            'expression': 'ev_integrate_mat.i.Y(mat_he.S, u)',
            'class': cb.CoefOne,
        },
        'Q1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, Pi1u, Pi2u)',
            'set_variables': [('Pi1u', 'pis_u', 'u'),
                              ('Pi2u', 'corrs_p', 'u')],
            'class': cb.CoefNonSym,
        },
        'Q2': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
            'set_variables': [('Pi1p', 'corrs_p', 'p'),
                              ('Pi1u', 'pis_u', 'u')],
            'class': cb.CoefNonSym,
        },
        'Q': {  # retardation stress, eq. (54)
            'requires': ['c.Q1', 'c.Q2'],
            'expression': 'c.Q1 - c.Q2',
            'class': cb.CoefEval,
        },
        'vol': {
            'status': 'auxiliary',
            'regions': ['Ym'] + [f'Yc{ch}' for ch in chs],
            'expression': 'd_volume.i.%s(u)',
            'class': cb.VolumeFractions,
        },
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'corrs_rs': {  # eq. (43)
            'requires': ['pis_u'],
            'ebcs': ['fixed_u', 'fixed_p'],
            'epbcs': periodic_all,
            'equations': {
                'balance_of_forces':
                """   dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                    - dw_biot.i.Ym(mat_he.BI, v, p)
                  = - dw_nonsym_elastic.i.Y(mat_he.A, v, Piu)""",
                'mass equilibrium':
                """ - dw_biot.i.Ym(mat_he.BI, u, q)
                -%e * dw_diffusion.i.Ym(mat_he.KH, q, p)
                    = dw_biot.i.Ym(mat_he.BI, Piu, q)""" % dt,
            },
            'set_variables': [('Piu', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_hp_rs',
            'solvers': {'ls': 'ls', 'nls': 'nls1', 'ts': None},
        },
        'corrs_p': {  #  particular response, eq. (45)
            'requires': ['press_m'],
            'ebcs': ['fixed_u', 'fixed_p'],
            'epbcs': periodic_all,
            'equations': {
                'balance_of_forces':
                """   dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                    - dw_biot.i.Ym(mat_he.BI, v, p)
                = - dw_lin_prestress.i.Y(mat_he.S, v)""",
                'mass equilibrium':
                """  - dw_biot.i.Ym(mat_he.BI, u, q)
                -%e * dw_diffusion.i.Ym(mat_he.KH, q, p)
                = %e * dw_diffusion.i.Ym(mat_he.KH, q, Pip)""" % (dt, dt),
            },
            'class': cb.CorrOne,
            'set_variables': [('Pip', 'press_m', 'p')],
            'save_name': 'corrs_hp_p',
            'solvers': {'ls': 'ls', 'nls': 'nls1', 'ts': None},
        },
        'press_m': {
            'variable': 'p',
            'class': CorrStatePressureM,
            'save_name': 'corrs_hp_press_m',
        },
    }

    for ich in chs:
        lab = '%d' % ich
        Yc = 'Yc' + lab
        pressurech = 'pressure' + lab
        pch = 'p' + lab
        corrsch = 'corrs_' + lab

        fields.update({
            pressurech: ('real', 'scalar', Yc, approx_p),
        })

        variables.update({
            pch: ('unknown field', pressurech),
            'q' + lab: ('test field', pressurech, pch),
            'Pip' + lab: ('parameter field', pressurech, pch),
            'Pi1p' + lab: ('parameter field', pressurech, '(set-to-None)'),
            'Pi2p' + lab: ('parameter field', pressurech, '(set-to-None)'),
            'P' + lab: ('parameter field', pressurech, '(set-to-None)'),
            'ls' + lab: ('unknown field', pressurech),
            'lv' + lab: ('test field', pressurech, 'ls' + lab),
        })

        epbcs.update({
            'periodic_px' + lab: (['Left', 'Right'],
                                  {'p%s.0' % lab: 'p%s.0' % lab},
                                  'match_x_plane'),
        })

        periodic_all_p = ['periodic_px' + lab, 'periodic_py' + lab]

        if dim == 3:
            epbcs.update({
                'periodic_py' + lab: (['Near', 'Far'],
                                      {'p%s.0' % lab: 'p%s.0' % lab},
                                      'match_y_plane'),
                'periodic_pz' + lab: (['Bottom', 'Top'],
                                      {'p%s.0' % lab: 'p%s.0' % lab},
                                      'match_z_plane'),
            })

            periodic_all_p += ['periodic_pz' + lab]
        else:
            epbcs.update({
                'periodic_py' + lab: (['Bottom', 'Top'],
                                      {'p%s.0' % lab: 'p%s.0' % lab},
                                      'match_y_plane'),
            })

        regions.update({
            Yc: 'cells of group %d' % (ich + 1),
            'Gamma' + lab: ('r.Yc%s *s r.Ym' % lab, 'facet', 'Ym'),
        })

        ename = 'fixed_p%s_%s_1' % (lab, lab)
        ebcs[ename] = ('Gamma' + lab, {'p.0': 1.})
        fixed_p_01 = [ename]

        chs2 = chs[:]
        chs2.remove(ich)
        for ich2 in chs2:
            lab2 = '%d' % ich2
            ename = 'fixed_p%s_%s_0' % (lab, lab2)
            ebcs[ename] = ('Gamma' + lab2, {'p.0': 0.})
            fixed_p_01.append(ename)

        lname = 'imv' + lab
        lcbcs[lname] = (Yc, {'ls%s.0' % lab: None}, None,
                        'integral_mean_value')

        coefs.update({
            'B%s_1' % lab: {
                'status': 'auxiliary',
                'requires': ['pis_u', corrsch],
                'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
                'set_variables': [('Pi1p', corrsch, 'p'),
                                  ('Pi1u', 'pis_u', 'u')],
                'class': cb.CoefNonSym,
            },
            # 'B%s_2' % lab: {
            #     'status': 'auxiliary',
            #     'requires': ['pis_u'],
            #     'expression': 'dw_lin_prestress.i.Yc%s(mat_he.BI, Pi1u)' % lab,
            #     'set_variables': [('Pi1u', 'pis_u', 'u')],
            #     'class': cb.CoefNonSym,
            # },
            'B%s_3' % lab: {
                'status': 'auxiliary',
                'requires': ['pis_u', corrsch],
                'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, Pi1u, Pi2u)',
                'set_variables': [('Pi1u', 'pis_u', 'u'),
                                  ('Pi2u', corrsch, 'u')],
                'class': cb.CoefNonSym,
            },
            'B' + lab: {  # The Biot poroelasticity tensor, eq. (52)
                # 'requires': ['c.B%s_%d' % (lab, ii + 1) for ii in range(3)],
                # 'expression': 'c.B%s_1 + c.B%s_2 - c.B%s_3' % ((lab,) * 3),
                'requires': [f'c.B{lab}_{ii}' for ii in [1, 3]] + [f'c.Phi{lab}'],
                'expression': f'c.B{lab}_1 - c.B{lab}_3 + c.Phi{lab} * {nonsym_eye[dim]}',
                'class': cb.CoefEval,
            },
            'C' + lab: {  # channel permeability, eq. (55)
                'requires': ['pis_p' + lab, 'corrs_eta' + lab],
                'expression': 'dw_diffusion.i.Yc%s(mat_he.KH, Pi1p%s, Pi2p%s)'\
                              % ((lab,) * 3),
                'set_variables': [('Pi1p' + lab,
                                  ('pis_p' + lab, 'corrs_eta' + lab), 'p' + lab),
                                  ('Pi2p' + lab,
                                  ('pis_p' + lab, 'corrs_eta' + lab), 'p' + lab)],
                'class': cb.CoefDimDim,
            },
            # 'Z%s_1' % lab: {
            #     'status': 'auxiliary',
            #     'requires': ['corrs_p'],
            #     'expression': 'dw_lin_prestress.i.Yc%s(mat_he.BI, Pi1u)' % lab,
            #     'set_variables': [('Pi1u', 'corrs_p', 'u')],
            #     'class': cb.CoefOne,
            # },
            'Z%s_1b' % lab: {
                'status': 'auxiliary',
                'requires': ['corrs_p'],
                'expression': f'-de_surface_ltr.i.Gamma{lab}(Pi1u)', #  - n^[3]
                'set_variables': [('Pi1u', 'corrs_p', 'u')],
                'class': cb.CoefOne,
            },
            'Z%s_2' % lab: {
                'status': 'auxiliary',
                'requires': [corrsch, 'corrs_p'],
                'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
                'set_variables': [('Pi1u', 'corrs_p', 'u'),
                                  ('Pi1p', corrsch, 'p')],
                'class': cb.CoefOne,
            },
            'Z%s_3' % lab: {
                'status': 'auxiliary',
                'requires': [corrsch, 'corrs_p', 'press_m'],
                'expression': 'dw_diffusion.i.Ym(mat_he.KH, Pi1p, Pi2p)',
                'set_variables': [('Pi1p', ('corrs_p', 'press_m'), 'p'),
                                  ('Pi2p', corrsch, 'p')],
                'class': cb.CoefOne,
            },
            # 'Z' + lab: {  # effective discharge, eq. (58)
            #     'requires': ['c.Z%s_%d' % (lab, ii + 1) for ii in range(3)],
            #     'expression': 'c.Z%s_1/%e + c.Z%s_2/%e + c.Z%s_3'\
            #                   % (lab, dt, lab, dt, lab),
            #     'class': cb.CoefEval,
            # },
            'Z' + lab: {  # effective discharge, eq. (58)
                'requires': [f'c.Z{lab}_{ii}' for ii in ['1b', 2, 3]],
                'expression': f'c.Z{lab}_1b/{dt} + c.Z{lab}_2/{dt} + c.Z{lab}_3',
                'class': cb.CoefEval,
            },
            # 'g%s_1' % lab: {  # effective discharge, eq. (58)
            #     'status': 'auxiliary',
            #     'requires': ['pis_p' + lab, 'press_' + lab],
            #     'expression': f'dw_diffusion.i.Yc{lab}(mat_he.KH, Pi2p{lab}, Pi1p{lab})',
            #     'set_variables': [('Pi1p' + lab, 'press_' + lab, 'p' + lab),
            #                       ('Pi2p' + lab, 'pis_p' + lab, 'p' + lab)],
            #     'class': cb.CoefDim,
            # },
            # 'g%s_2' % lab: {  # effective discharge, eq. (58)
            #     'status': 'auxiliary',
            #     'requires': ['pis_p' + lab, 'press_' + lab],
            #     'expression': f'dw_diffusion.i.Yc{lab}(mat_he.dK, Pi2p{lab}, Pi1p{lab})',
            #     'set_variables': [('Pi1p' + lab, 'press_' + lab, 'p' + lab),
            #                       ('Pi2p' + lab, 'pis_p' + lab, 'p' + lab)],
            #     'class': cb.CoefDim,
            # },
            # 'g%s_3' % lab: {  # effective discharge, eq. (58)
            #     'status': 'auxiliary',
            #     'requires': ['pis_p' + lab, 'corrs_p' + lab],
            #     'expression': f'dw_diffusion.i.Yc{lab}(mat_he.KH, Pi2p{lab}, Pi1p{lab})',
            #     'set_variables': [('Pi1p' + lab, 'corrs_p' + lab, 'p' + lab),
            #                       ('Pi2p' + lab, 'pis_p' + lab, 'p' + lab)],
            #     'class': cb.CoefDim,
            # },
            # 'g' + lab: {  # effective discharge, eq. (58)
            #     'requires': [f'c.g{lab}_1', f'c.g{lab}_2', f'c.g{lab}_3'],
            #     'expression': f'c.g{lab}_1 + c.g{lab}_2 + c.g{lab}_3',
            #     'class': cb.CoefEval,
            # },
            # 'g%s' % lab: {  # effective discharge, eq. (58)
            #     'requires': ['pis_p' + lab, 'press_' + lab, 'corrs_p' + lab],
            #     'expression': """dw_diffusion.i.Yc%s(mat_he.KH, Pi1p%s, Pi2p%s)"""\
            #                 % ((lab,) * 3),
            #     'set_variables': [('Pi1p' + lab, ('press_' + lab, 'corrs_p' + lab),
            #                        'p' + lab),
            #                       ('Pi2p' + lab, 'pis_p' + lab, 'p' + lab)],
            #     'class': cb.CoefDim,
            # },
            'Phi' + lab: {
                'status': 'auxiliary',
                'requires': ['c.vol'],
                'expression': f'c.vol["fraction_Yc{lab}"]',
                'class': cb.CoefEval,
            },
        })

        requirements.update({
            corrsch: {  # eq. (44)
                'requires': [],
                'ebcs': ['fixed_u'] + fixed_p_01,
                'epbcs': periodic_all,
                'equations': {
                    'balance_of_forces':
                    """   dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                        - dw_biot.i.Ym(mat_he.BI, v, p)
                        = dw_lin_prestress.i.Yc%s(mat_he.BI, v)""" % lab,
                    'mass equilibrium':
                    """ - dw_biot.i.Ym(mat_he.BI, u, q)
                    -%e * dw_diffusion.i.Ym(mat_he.KH, q, p)
                        = 0""" % dt,
                },
                'class': cb.CorrOne,
                'save_name': 'corrs_hp_' + lab,
                'solvers': {'ls': 'ls', 'nls': 'nls1', 'ts': None},
            },
            'pis_p' + lab: {
                'variables': [pch],
                'class': cb.ShapeDim,
            },
            'corrs_eta' + lab: {  # channel flow correctors, eq. (46)
                'requires': ['pis_p' + lab],
                'epbcs': periodic_all_p,
                'ebcs': [],
                'lcbcs': [lname],
                'equations': {
                    'eq':
                    """   dw_diffusion.i.Yc%s(mat_he.KH, q%s, p%s)
                        + dw_dot.i.Yc%s(q%s, ls%s)
                        =
                        - dw_diffusion.i.Yc%s(mat_he.KH, q%s, Pip%s)"""\
                        % ((lab,) * 9),
                    'eq_imv':
                        'dw_dot.i.Yc%s(lv%s, p%s) = 0' % ((lab,) * 3),
                },
                'class': cb.CorrDim,
                'set_variables': [('Pip' + lab, 'pis_p' + lab, pch)],
                'save_name': 'corrs_hp_eta_' + lab,
                'solvers': {'ls': 'ls', 'nls': 'nls2', 'ts': None},
            },
            # 'corrs_p' + lab: {  # particular response, eq. (47)
            #     'requires': ['press_' + lab],
            #     'ebcs': [],
            #     'epbcs': periodic_all_p,
            #     'lcbcs': [lname],
            #     'equations': {
            #         'eq':
            #         """   dw_diffusion.i.Yc%s(mat_he.KH, q%s, p%s)
            #             + dw_dot.i.Yc%s(q%s, ls%s)
            #             =
            #             dw_diffusion.i.Yc%s(mat_he.dK, q%s, Pip%s)
            #             - dw_diffusion.i.Yc%s(mat_he.KH, q%s, Pip%s)"""\
            #             % ((lab,) * 12),
            #         'eq_imv':
            #             'dw_dot.i.Yc%s(lv%s, p%s) = 0' % ((lab,) * 3),

            #     },
            #     'class': cb.CorrOne,
            #     'set_variables': [('Pip' + lab, 'press_' + lab, pch)],
            #     'save_name': 'corrs_hp_p_' + lab,
            #     'solvers': {'ls': 'ls', 'nls': 'nls2', 'ts': None},
            # },
            'press_' + lab: {
                'variable': pch,
                'class': CorrStatePressureCh,
                'save_name': 'corrs_hp_press_' + lab,
            },
        })

        for ich2 in chs:
            lab2 = f'{ich2}'
            lab12 = lab + lab2
            corrsch2 = 'corrs_' + lab2

            coefs.update({
                # f'G{lab12}_3': {
                #     'status': 'auxiliary',
                #     'requires': [corrsch2],
                #     'expression': 'dw_lin_prestress.i.Yc%s(mat_he.BI, Pi1u)' % lab,
                #     'set_variables': [('Pi1u', corrsch2, 'u')],
                #     'class': cb.CoefOne,
                # },
                f'G{lab12}_3b': {
                    'status': 'auxiliary',
                    'requires': [corrsch2],
                    'expression': f'-de_surface_ltr.i.Gamma{lab}(Pi1u)', #  - n^[3]
                    'set_variables': [('Pi1u', corrsch2, 'u')],
                    'class': cb.CoefOne,
                },
                f'G{lab12}_2': {
                    'status': 'auxiliary',
                    'requires': [corrsch, corrsch2],
                    'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
                    'set_variables': [('Pi1u', corrsch, 'u'),
                                      ('Pi1p', corrsch2, 'p')],
                    'class': cb.CoefOne,
                },
                f'G{lab12}_1': {
                    'status': 'auxiliary',
                    'requires': [corrsch, corrsch2],
                    'expression': 'dw_diffusion.i.Ym(mat_he.KH, Pi1p, Pi2p)',
                    'set_variables': [('Pi1p', corrsch, 'p'),
                                      ('Pi2p', corrsch2, 'p')],
                    'class': cb.CoefOne,
                },
                # f'G{lab12}': {  # perfusion coefficient , eq. (57)
                #     'requires': [f'c.G{lab12}_{ii + 1}' for ii in range(3)],
                #     'expression': f'c.G{lab12}_1/{dt} + c.G{lab12}_2/{dt} + c.G{lab12}_3',
                #     'class': cb.CoefEval,
                # },
                f'G{lab12}': {  # perfusion coefficient , eq. (57)
                    'requires': [f'c.G{lab12}_{ii}' for ii in [1, 2, '3b']],
                    'expression': f'c.G{lab12}_1 + c.G{lab12}_2/{dt} + c.G{lab12}_3b/{dt}',
                    'class': cb.CoefEval,
                },
            })

    solvers = {
        'ls': ('ls.mumps', {
            'memory_relaxation': 50,
        }),
        'nls2': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-16,
            'eps_r': 1e-3,
            'problem': 'nonlinear',
        }),
        'nls1': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-6,
            'eps_r': 1e-3,
            'problem': 'nonlinear',
        }),

    }

    return locals()
