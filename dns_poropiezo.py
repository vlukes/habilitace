import os.path as osp
import numpy as nm
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.base.base import Struct, debug
from sfepy.discrete.fem.mesh import Mesh
from sfepy.solvers.ts import TimeStepper
from sfepy.mesh.mesh_generators import tiled_mesh1d
import sfepy.discrete.fem.periodic as per
from sfepy.postprocess.probes_vtk import Probe
from scipy.io import savemat

wdir = osp.dirname(__file__)


def mhook(mtx, pb, call_mode=None):
    from sfepy.discrete.common.dof_info import expand_nodes_to_equations

    if call_mode == 'basic' or call_mode == 'residual':
        variables = pb.get_variables()
        var = variables['u']
        field = var.field
        region = pb.domain.regions['Right_ef']
        vertices = field.get_dofs_in_region(region, merge=True)
        dofs = expand_nodes_to_equations(vertices, ['u.0'], var.dofs)
        eq = var.eq_map.eq[dofs]
        ia = nm.where(eq >= 0)[0]
        seq = eq[ia[1:]]
        m = eq[ia[0]]
        stiffness = 0.1e9
    if call_mode == 'basic':
        mtx[m, m] += stiffness * len(seq)
        for s in seq:
            mtx[m, s] += -stiffness
            mtx[s, m] += -stiffness
            mtx[s, s] += stiffness
    elif call_mode == 'residual':
        st = variables.get_state(True)
        mtx[m] += stiffness * len(seq) * st[m]
        for s in seq:
            mtx[m] += -stiffness * st[s]
            mtx[s] += stiffness * (st[s] - st[m])

    return mtx


def gen_tiled_mesh(mesh, grid=None, scale=1.0, eps=1e-2):
    bbox = mesh.get_bounding_box()

    if grid is None:
        iscale = max(int(1.0 / scale), 1)
        grid = [iscale] * mesh.dim

    conn = mesh.get_conn(mesh.descs[0])
    mat_ids = mesh.cmesh.cell_groups

    coors = mesh.coors
    ngrps = mesh.cmesh.vertex_groups
    nrep = nm.prod(grid)

    print('repeating %s ...' % grid)
    nblk = 1
    mat_ids_out = []
    for ii, gr in enumerate(grid):
        conn, coors, ngrps = tiled_mesh1d(conn, coors, ngrps,
                                          ii, gr, bbox.transpose()[ii],
                                          eps=eps)
        nblk *= gr

    for ii in range(nrep):
        mat_ids0 = mat_ids.copy()
        mat_ids_out.append(mat_ids0)

    print('...done')

    mat_ids = nm.array(mat_ids_out).flatten()
    mesh_out = Mesh.from_data('tiled mesh', coors * scale, ngrps,
                              [conn], [mat_ids], [mesh.descs[0]])

    return mesh_out


def post_process_final(pb):
    print('>>> final post-processing')
    cache = pb.cache
    out = {}

    out_data = cache['out_data']
    for k, field in enumerate(out_data[0]):
        val = []
        for step in out_data[1:]:
            val.append(step[k])

        out[field] = nm.array(val)

    out.update({k: nm.array(v) for k, v in cache.items()
                if k.startswith('fluxw_')})

    out.update({k: nm.array(v) for k, v in cache.items()
                if k.startswith('S_')})

    out.update({k: nm.array(v) for k, v in cache.items()
                if k.startswith('w_avg_')})

    out.update({'niter': nm.array(cache['niter'])})

    fname = osp.join(pb.conf.options.get('output_dir', '.'),
                     f'poropiezo_data_dns{pb.conf.flag}.mat')

    savemat(fname, out)


def post_process(out, pb, state, extend=False):
    print('>>> post-processing')
    conf = pb.conf
    cache = pb.cache

    sc_u = conf.scale_u
    sc_r = conf.scale_r
    sc_p = conf.scale_p
    sc_w = conf.scale_w

    nv = pb.domain.mesh.n_nod
    nc = pb.domain.mesh.n_el

    tstep = conf.tstep

    out = {}

    idxs = pb.domain.regions['Omega'].entities[0]
    u_full = nm.zeros((nv, 3), dtype=nm.float64)
    u_full[idxs, :] = state['u'].reshape((-1, 3)) * sc_u

    idxs = pb.domain.regions['Omega_p'].entities[0]
    r_full = nm.zeros((nv, 1), dtype=nm.float64)
    r_full[idxs, :] = state['r'][:, None] * sc_r

    out['u'] = Struct(name='output_data', mode='vertex',
                      var_name='u', data=u_full)
    out['r'] = Struct(name='output_data', mode='vertex',
                      var_name='r', data=r_full)

    bolus = get_bolus(tstep, pb.domain.mesh.coors, problem=pb)[0] * sc_r
    bolus = nm.vstack([bolus * 0, bolus * 0, bolus]).T * conf.phi_ampl[0]
    out['bolus'] = Struct(name='output_data', mode='vertex', data=bolus)

    idxs = pb.domain.regions['Omega_f'].entities[0]
    p_full = nm.zeros((nv, 1), dtype=nm.float64)
    p_full[idxs, :] = state['p'][:, None] * sc_p

    idxs = pb.domain.regions['Omega_f'].entities[0]
    w_full = nm.zeros((nv, 3), dtype=nm.float64)
    w_full[idxs, :] = state['w'][:(idxs.shape[0] * 3)].reshape((-1, 3)) * sc_w

    out['p'] = Struct(name='output_data', mode='vertex',
                      var_name='p', data=p_full)
    out['w'] = Struct(name='output_data', mode='vertex',
                      var_name='w', data=w_full)

    idxs = pb.domain.regions['Omega_f'].entities[-1]

    grad_p = pb.evaluate('ev_grad.i2.Omega_f(p)', mode='el_avg')
    grad_p_full = nm.zeros((nc, 1, 3, 1), dtype=nm.float64)
    grad_p_full[idxs, ...] = grad_p.reshape((-1, 1, 3, 1)) * sc_p
    out['grad_p'] = Struct(name='output_data',
                           mode='cell',
                           dofs=None,
                           var_name='p',
                           data=grad_p_full)

    mesh = pb.domain.mesh
    fname = f'{pb.ofn_trunk}{conf.flag}_{tstep.step:03d}.vtk'
    mesh.write(osp.join(conf.options.get('output_dir', '.'), fname), out=out)

    probe = Probe(out, mesh)
    eps0 = pb.conf.bbox[1][-1]
    n_points = 100

    p1, p2 = nm.array([[0.0, 0.1, 0.3], [0.1 / eps0, 0.1, 0.3]]) * eps0
    probe.add_line_probe('line1', p1, p2, n_points)
    prb_u = probe('line1', 'u')

    p1, p2 = nm.array([[0.0, 0.1, 0.14], [0.1 / eps0, 0.1, 0.14]]) * eps0
    probe.add_line_probe('line2', p1, p2, n_points)
    prb_r0 = probe('line2', 'r')

    p1, p2 = nm.array([[0.0, 0.1, 0.86], [0.1 / eps0, 0.1, 0.86]]) * eps0
    probe.add_line_probe('line3', p1, p2, n_points)
    prb_r1 = probe('line3', 'r')

    p1, p2 = nm.array([[0.0, 0.5, 0.5], [0.1 / eps0, 0.5, 0.5]]) * eps0
    probe.add_line_probe('line4', p1, p2, n_points)
    prb_p = probe('line4', 'p')
    prb_w = probe('line4', 'w')

    lc = prb_u[0]
    cx = p1[0] * (1 - lc) + p2[0] * lc
    bolus = get_bolus(tstep, nm.array([cx, cx * 0, cx * 0]).T,
                      problem=pb)[0] * pb.conf.phi_ampl[0] * sc_r

    if 'out_data' not in cache:
        cache['out_data'] = [('time', 'bolus', 'lc',
                              'displ', 'potential_1', 'potential_2',
                              'pressure', 'velocity')]

    out_data = cache['out_data']
    out_data.append((tstep.time, bolus, lc,
                     prb_u[1], prb_r0[1], prb_r1[1], prb_p[1], prb_w[1]))
    # boundary flux
    pp_regs = [('Left', 0, -1), ('Right', 0, 1), ('Middle', 0, 1),
               ('Left1', 0, -1), ('Right1', 0, 1)]

    pvars = pb.get_variables()
    pvars['W0'].set_data(state['w'])

    for reg, dir, mul in pp_regs:
        w = pb.evaluate(f'ev_integrate.i2.{reg}_f(W0)', mode='el_avg') * sc_w
        S = pb.evaluate(f'ev_volume.i2.{reg}_f(W0)', mode='el_avg')
        Q = (w[..., dir, 0] * S[..., 0, 0]).sum() * tstep.dt * mul

        key = 'fluxw_%s' % reg
        if key not in cache:
            cache[key] = []
        cache[key].append(Q)

        key = 'S_%s' % reg
        if key not in cache:
            cache[key] = []
        cache[key].append(S)

        w_avg = nm.average(w, axis=0).squeeze() * mul
        key = 'w_avg_%s' % reg
        if key not in cache:
            cache[key] = []
        cache[key].append(w_avg[dir])

    return out


def time_stepping(pb):
    pb.cache = cache = {}
    is_nonlinear = pb.conf.is_nonlinear
    vars = pb.get_variables()
    tstep = pb.conf.tstep

    state_list = ['u', 'r', 'p', 'w']
    inc_list = [f'{k}0' for k in state_list]

    cache['niter'] = []
    cache['state'] = state = {k: nm.zeros((vars[k[0]].n_dof,),
                                          dtype=nm.float64)
                              for k in (state_list + inc_list)}

    coors0 = pb.domain.mesh.coors.copy()

    sc_u = pb.conf.scale_u

    out = []
    for step, time in tstep:
        print('##################################################')
        print(f'  step: {step}')
        print('##################################################')

        for k in state_list:
            vars[f'{k.upper()}0'].set_data(state[k])

        norm_state = {k: 1. for k in state_list}

        iiter = 1
        norm_i0 = {}
        next_iteration = True
        while next_iteration:

            if is_nonlinear:
                print('##### mesh update')
                new_coors = coors0 + state['u'].reshape(coors0.shape) * sc_u
                pb.set_mesh_coors(new_coors, update_fields=True,
                                  clear_all=True)

            yield pb, out

            new_state = out[-1][1].get_state_parts()

            for k in state_list:
                d0 = state[k] - state[f'{k}0']
                norm_state[k] = nm.linalg.norm(new_state[k] - d0)
                if iiter == 1:
                    norm_i0[k] = nm.linalg.norm(new_state[k])
                if nm.all(norm_i0[k] > 0):
                    norm_state[k] /= norm_i0[k]

                state[k][:] = state[f'{k}0'] + new_state[k]

            if is_nonlinear:
                print('--------------------------------------------------')
                print(f'  iter: {iiter}')
                print(f'  u: {norm_state["u"]}')
                print(f'  p: {norm_state["p"]}')
                print(f'  u_max: {nm.max(nm.abs(new_state["u"]))}')
                print('--------------------------------------------------')

                if norm_state['p'] < 1e-2:
                    next_iteration = False

                iiter += 1

                if iiter > 20:
                    print('maximal number of iterations!!!')
                    debug()

            else:
                next_iteration = False

            yield None

        for k in state_list:
            state[f'{k}0'][:] = state[k]

        cache['niter'].append(iiter)

        post_process(None, pb, state)

    post_process_final(pb)


def press_nbc(ts, coors, mode=None, problem=None, **kwargs):
    if not (mode == 'qp'):
        return

    conf = problem.conf
    p_bar = conf.p_bar
    val = nm.eye(3) * p_bar * conf.scale_w * conf.scale_p

    return {'dval': nm.tile(val, (coors.shape[0], 1, 1))}


def press_dbc(ts, coors, problem=None, **kwargs):
    ts = problem.conf.tstep
    p_bar = problem.conf.p_bar if ts.step == 0 else 0
    return nm.ones((coors.shape[0], 1), dtype=nm.float64) * p_bar


def get_surf_regions(regs):
    surfs = ['Left', 'Right', 'Near', 'Far', 'Bottom', 'Top']
    out = {}

    for r in regs:
        out.update(
            {f'{s}_{r}': (f'r.{s} *s r.Omega_{r}', 'facet') for s in surfs}
        )

    return out


def get_bolus_cos(ts, coors):
    L = 0.1  # domain length
    # c = 5    # wave speed [L/s]
    # X = 0.6  # wave length [L]
    c = 3  # wave speed [L/s]
    X = 0.6  # wave length [L]

    k = 2 * nm.pi / (X * L)
    t, dt = ts.time, ts.dt

    n = coors.shape[0]
    x = coors[:, 0].reshape((1, n))
    if isinstance(t, nm.ndarray):
        t = t[:, None]

    out = nm.zeros((1, n), dtype=nm.float64)
    dout = nm.zeros((1, n), dtype=nm.float64)

    ang = k * (x - c * L * t) + nm.pi

    psi = 0.5 * (1 + nm.cos(ang))
    lv = x - c * L * t < 0
    out[lv] = psi[lv]
    dout[lv] = 0.5 * c * L * k * nm.sin(ang[lv])

    return out, dout * dt


def get_bolus(ts, coors, **kwargs):
    pb = kwargs.get('problem')
    ts = pb.conf.tstep

    return get_bolus_cos(ts, coors)


def get_bolus_bc(ts, coors, **kwargs):
    pb = kwargs.get('problem')
    ts = pb.conf.tstep

    out, _ = get_bolus(ts, coors, **kwargs)
    out = out.T * pb.conf.phi_ampl[0]

    pb.cache[f'bolus_{ts.step}'] = out.copy()
    key = f'bolus_{ts.step - 1}'
    if key in pb.cache:
        out -= pb.cache[f'bolus_{ts.step - 1}']

    return out


def define(
        mode='1D',
        is_nonlinear=True,
        t_end=1,
        t_nstep=50,
        output_dir='output',
        N=20,
        phi_ampl=nm.array([1, 0]) * 4e5,
        p_bar=0,
    ):

    scale_u = 1e-5  # u = scale_u * \bar u
    scale_r = 1e5  # r = scale_r * \bar r
    scale_p = 1e1  # p = scale_p * \bar p
    scale_w = 1e4  # w = scale_w * \bar w

    phi_ampl *= 1 / scale_r
    p_bar *= 1 / scale_p

    filename_mesh = osp.join(wdir, 'meshes', f'mesh_poropiezo_dns_{N}x1x1.vtk')

    nl_flag = {True: 'nl', False: 'l'}[is_nonlinear]

    flag = f'_1D_{nl_flag}_{N}'

    if not osp.exists(filename_mesh):
        print(f'generating {N}x1x1 mesh: {filename_mesh}')
        filename_mesh_micro = osp.join(wdir, 'mesh_micro.vtk')
        mesh0 = Mesh.from_file(filename_mesh_micro)
        mesh1 = gen_tiled_mesh(mesh0, grid=[N, 1, 1], scale=0.1/N)
        mesh1.write(filename_mesh)

    tstep = TimeStepper(0.0, t_end, n_step=t_nstep)

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'potential': ('real', 'scalar', 'Omega_p', 1),
        'velocity': ('real', 'vector', 'Omega_f', 2),
        'pressure': ('real', 'scalar', 'Omega_f', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'r': ('unknown field', 'potential'),
        's': ('test field', 'potential', 'r'),
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'w': ('unknown field', 'velocity'),
        'z': ('test field', 'velocity', 'w'),
        'U0': ('parameter field', 'displacement', 'u'),
        'R0': ('parameter field', 'potential', 'r'),
        'P0': ('parameter field', 'pressure', 'p'),
        'W0': ('parameter field', 'velocity', 'w'),
    }

    bbox = [[0, 0, 0], [0.1, 0.1/N, 0.1/N]]
    delta = 1e-8
    regions = {
        'Omega_e': 'cells of group 1',
        'Omega_p': 'cells of group 2',
        'Omega_f': 'cells of group 3',
        'Omega_m': ('r.Omega_e +v r.Omega_p', 'cell'),
        'Omega_c1': 'cells of group 4',
        'Omega_c2': 'cells of group 5',
        'Omega_c': ('r.Omega_c1 +v r.Omega_c2', 'cell'),
        'Omega_s': ('r.Omega_m +v r.Omega_c', 'cell'),
        'Omega': ('r.Omega_s +v r.Omega_f', 'cell'),
        'Gamma_c1': ('r.Omega_m *s r.Omega_c1', 'facet', 'Omega_c1'),
        'Gamma_c2': ('r.Omega_m *s r.Omega_c2', 'facet', 'Omega_c2'),
        'Left': (f'vertices in (x < {bbox[0][0] + delta})', 'facet'),
        'Right': (f'vertices in (x > {bbox[1][0] - delta})', 'facet'),
        'Near': (f'vertices in (y < {bbox[0][1] + delta})', 'facet'),
        'Far': (f'vertices in (y > {bbox[1][1] - delta})', 'facet'),
        'Bottom': (f'vertices in (z < {bbox[0][2] + delta})', 'facet'),
        'Top': (f'vertices in (z > {bbox[1][2] - delta})', 'facet'),
        'Gamma_inter': ('r.Omega_s *s r.Omega_f', 'facet', 'Omega_f'),
        'Left_f': ('r.Left *s r.Omega_f', 'facet'),
        'Right_f': ('r.Right *s r.Omega_f', 'facet'),
        'Right_e': ('r.Right *s r.Omega_e', 'facet'),
        'Right_ef': ('r.Right_e +s r.Right_f', 'facet'),
        'OL': ('vertices in (x < 0.0501)', 'cell'),
        'OR': ('vertices in (x > 0.0499)', 'cell'),
        'Middle': ('r.OL *v r.OR', 'facet', 'OL'),
        'Middle_f': ('r.Middle *s r.Omega_f', 'facet'),
        'OL1': ('vertices in (x < %e)' % (0.1/N + 1e-4), 'cell'),
        'OR1': ('vertices in (x > %e)' % (0.1/N - 1e-4), 'cell'),
        'Left1': ('r.OL1 *v r.OR1', 'facet', 'OR1'),
        'Left1_f': ('r.Left1 *s r.Omega_f', 'facet'),
        'OL2': ('vertices in (x < %e)' % (0.1/N*(N - 1) + 1e-4), 'cell'),
        'OR2': ('vertices in (x > %e)' % (0.1/N*(N - 1) - 1e-4), 'cell'),
        'Right1': ('r.OL2 *v r.OR2', 'facet', 'OL2'),
        'Right1_f': ('r.Right1 *s r.Omega_f', 'facet'),
        'EdgeY': ('r.Left *v r.Bottom', 'vertex'),
        'EdgeZ': ('r.Left *v r.Near', 'vertex'),
    }

    regions.update(get_surf_regions(['s', 'p']))

    D_piezo = nm.array([[6.0, 3.72, 3.83, 0, 0, 0],
                        [3.72, 6.0, 3.83, 0, 0, 0],
                        [3.83, 3.83, 20.3, 0, 0, 0],
                        [0, 0, 0, 1.23, 0, 0],
                        [0, 0, 0, 0, 1.23, 0],
                        [0, 0, 0, 0, 0, 1.23]]) * 1e9
    g_piezo = nm.array([[0, 0, 0, 0, 0.01, 0],
                        [0, 0, 0, 0, 0, 0.01],
                        [-0.09, -0.09, 5.91, 0, 0, 0]])
    d_piezo = nm.array([[18, 0, 0],
                        [0, 18, 0],
                        [0, 0, 255.3]]) * 8.854*1e-12

    D_elast = stiffness_from_youngpoisson(3, 0.02e9, 0.49)
    D_cond = stiffness_from_youngpoisson(3, 200e9, 0.25)  # Cu
    D_fluid = D_elast * 1e-2  # artificial stiffness in fluid domain

    sc_D = scale_u**2

    materials = {
        'matrix': ({
            'D': {
                'Omega_p': D_piezo * sc_D,
                'Omega_e': D_elast * sc_D,
                'Omega_c': D_cond * sc_D,
                'Omega_f': D_fluid * sc_D,
            }
        },),
        'piezo': ({
            'g': g_piezo * scale_r * scale_u,
            'd': d_piezo * scale_r**2,
        },),
        'fluid': ({
            'sc_pu': scale_p * scale_u,
            'sc_pw': scale_p * scale_w,
            'eta_ww': 8.9e-4 * scale_w**2,
            'eta_wu': 8.9e-4 * scale_w * scale_u,
            'eta_uu': 8.9e-4 * scale_u**2,
        },),
        'press_nbc': 'press_nbc',
    }

    options = {
        'output_dir': output_dir,
        'parametric_hook': 'time_stepping',
        'matrix_hook': 'mhook',
    }

    functions = {
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
        'time_phi': (get_bolus_bc,),
        'press_nbc': (press_nbc,),
        'press_dbc': (press_dbc,),
    }

    ebcs = {
        'fix_ux': ('Left', {'u.0': 0.0}),
        'fix_uy': ('EdgeZ', {'u.1': 0.0}),
        'fix_uz': ('EdgeY', {'u.2': 0.0}),
        'fix_w': ('Gamma_inter', {'w.all': 0.0}),
        'fix_p_left': ('Left_f', {'p.0': 0.0}),
        'fix_p_right': ('Right_f', {'p.0': 'press_dbc'}),
        'cond1': ('Gamma_c1', {'r.0': 'time_phi'}),
        'cond2': ('Gamma_c2', {'r.0': 0.0}),
    }

    epbcs = {
        'per_u_y': (['Near', 'Far'], {'u.all': 'u.all'}, 'match_y_plane'),
        'per_u_z': (['Bottom', 'Top'], {'u.all': 'u.all'}, 'match_z_plane'),
        'per_r_y': (['Near_s', 'Far_s'], {'r.0': 'r.0'}, 'match_y_plane'),
        'per_r_z': (['Bottom_s', 'Top_s'], {'r.0': 'r.0'}, 'match_z_plane'),
    }

    integrals = {
        'i2': 2,
        'i3': 3,
    }

    solvers = {
        'ls': ('ls.mumps', {}),
        'newton': ('nls.newton', {
            'i_max': 10,
            'eps_a': 1e-2,
            'eps_r': 2e-3,
            'problem': 'nonlinear',
        }),
    }

    idt = 1. / tstep.dt

    equations = {
        'balance_of_forces': f"""
            dw_lin_elastic.i2.Omega(matrix.D, v, u)
          - dw_piezo_coupling.i2.Omega_p(piezo.g, v, r)
          + de_div_grad.i2.Omega_f(fluid.eta_wu, v, w)
  + {idt} * de_div_grad.i2.Omega_f(fluid.eta_uu, v, u)
          - dw_stokes.i2.Omega_f(fluid.sc_pu, v, p)
          =
          - dw_lin_elastic.i2.Omega(matrix.D, v, U0)
          + dw_piezo_coupling.i2.Omega_p(piezo.g, v, R0)
          - de_div_grad.i2.Omega_f(fluid.eta_wu, v, W0)
          + dw_stokes.i2.Omega_f(fluid.sc_pu, v, P0)""",
        'piezo': """
          - dw_piezo_coupling.i2.Omega_p(piezo.g, u, s)
          - dw_diffusion.i2.Omega_p(piezo.d, s, r)
          =
            dw_piezo_coupling.i2.Omega_p(piezo.g, U0, s)
          + dw_diffusion.i2.Omega_p(piezo.d, s, R0)""",
        'fluid_velocity': f"""
            de_div_grad.i3.Omega_f(fluid.eta_ww, z, w)
  + {idt} * de_div_grad.i3.Omega_f(fluid.eta_wu, z, u)
          - dw_stokes.i3.Omega_f(fluid.sc_pw, z, p)
          =
          - dw_surface_ltr.i2.Right_f(press_nbc.dval, z)
          - de_div_grad.i3.Omega_f(fluid.eta_ww, z, W0)
          + dw_stokes.i3.Omega_f(fluid.sc_pw, z, P0)""",
        'pressure': f"""
          - dw_stokes.i3.Omega_f(fluid.sc_pw, w, q)
  - {idt} * dw_stokes.i3.Omega_f(fluid.sc_pu, u, q)
          =
          + dw_stokes.i3.Omega_f(fluid.sc_pw, W0, q)""",
    }

    return locals()
