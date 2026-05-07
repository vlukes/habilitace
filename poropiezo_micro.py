import os.path as osp
import numpy as nm
from copy import deepcopy
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.homogenization.utils import coor_to_sym, define_box_regions, iter_sym
import sfepy.homogenization.coefs_base as cb
from sfepy.discrete.fem.mesh import Mesh
from sfepy.discrete.fem.periodic import match_grid_plane
import sfepy.base.multiproc as multiproc
from sfepy.base.base import Struct, get_default

mp_module, _ = multiproc.get_multiproc()
multiproc_dependecies = mp_module.get_dict('dependecies', clear=True)

wdir = osp.dirname(__file__)

# Y. Koutsawa et al. / Mechanics Research Communications 37 (2010) 489-494
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
D_cond = stiffness_from_youngpoisson(3, 200e9, 0.25)  # Cu?

periodic_cache = {}


def match_plane(coor1, coor2, d, di):
    key = '%s_%d' % (d, coor1.shape[0])
    if key not in periodic_cache:
        periodic_cache[key] = match_grid_plane(coor1, coor2, di)

    return periodic_cache[key]


def match_x_plane(coor1, coor2):
    return match_plane(coor1, coor2, 'x', 0)


def match_y_plane(coor1, coor2):
    return match_plane(coor1, coor2, 'y', 1)


def match_z_plane(coor1, coor2):
    return match_plane(coor1, coor2, 'z', 2)


def get_periodic_bc(var_tab, dim=3, dim_tab=None, regions=None):
    if dim_tab is None:
        dim_tab = {'x': ['left', 'right'],
                   'z': ['bottom', 'top'],
                   'y': ['near', 'far']}

    periodic = {}
    epbcs = {}

    for ivar, reg in var_tab:
        periodic['per_%s' % ivar] = pers = []
        for idim in 'xyz'[0:dim]:
            key = 'per_%s_%s' % (ivar, idim)
            regs = ['%s_%s' % (reg, ii) for ii in dim_tab[idim]]
            if regions is not None:
                if regs[0] not in regions or regs[1] not in regions:
                    regs = []

            if len(regs) > 0:
                epbcs[key] = (regs, {'%s.all' % ivar: '%s.all' % ivar},
                              'match_%s_plane' % idim)
                pers.append(key)

    return epbcs, periodic


def data_to_struct(data):
    out = {}
    for k, v in data.items():
        out[k] = Struct(name='output_data',
                        mode='cell' if v[2] == 'c' else 'vertex',
                        data=v[0],
                        var_name=v[1],
                        dofs=None)

    return out


def get_periodic_regions(reg, label=None, mesh_data=None, eps=1e-9):
    if label is None:
        label = reg

    dim_tab = [['Left', 'Right'], ['Near', 'Far'], ['Bottom', 'Top']]

    if mesh_data is not None:
        mat_id, mesh = mesh_data
        idxs = mesh.cmesh.cell_groups == mat_id
        coors = mesh.coors[mesh.get_conn(mesh.descs[0])[idxs].ravel()]
        bbox = mesh.get_bounding_box()
        cmin, cmax = nm.min(coors, axis=0), nm.max(coors, axis=0)
        cmin, cmax = abs(cmin - bbox[0, :]), abs(cmax - bbox[1, :])
        bbox_list = []
        for dim in range(3):
            if cmin[dim] < eps and cmax[dim] < eps:
                bbox_list += dim_tab[dim]
    else:
        bbox_list = dim_tab[0] + dim_tab[1] + dim_tab[2]

    out = {}
    for k in bbox_list:
        out[f'{label}_{k.lower()}'] = (f'r.{reg} *s r.{k}', 'facet')

    return out


def recovery_micro(pb, corrs, macro):
    eps0 = macro['eps0']
    mesh = pb.domain.mesh
    dim = mesh.dim

    regions = pb.domain.regions

    map_flag = macro['press'].shape[0] > 1
    Ys_map = regions['Ys'].get_entities(0)

    ncond = macro['phi'].shape[-1]
    gl = '_' + '_'.join(pb.conf.filename_coefs.split('_')[2:])

    mvar = pb.create_variables(['u', 'r', 'svar', 'p', 'w', 'uf'])

    if map_flag:
        Yp_map = regions['Yp'].get_entities(0)
        Yf_map = mvar['w'].field.vertex_remap_i
        press_mac_s = macro['press'][Ys_map, :, 0]
        press_mac_p = macro['press'][Yp_map, :, 0]
        press_mac_f = macro['press'][Yf_map, :, 0]
        phi_mac_s = macro['phi'][Ys_map, 0, :]
        phi_mac_p = macro['phi'][Yp_map, 0, :]
        strain_mac_s = macro['strain'][Ys_map, :, 0]
        strain_mac_p = macro['strain'][Yp_map, :, 0]
        displ_mac_s = macro['displ'][Ys_map, :, 0]
        press_mac_grad_f = macro['pressg'][Yf_map, 0, :]
    else:
        from sfepy.base.base import debug; debug()
        # press_mac_s = press_mac_p = press_mac_f = macro['press']
        # phi_mac_s = phi_mac_p = macro['phi']
        # strain_mac_s = strain_mac_p = macro['strain'].T
        # displ_mac_s = macro['displ'].T
        # press_mac_grad_f = macro['pressg'].T

    u1 = -corrs['corrs_p' + gl]['u'] * press_mac_s
    phi = -corrs['corrs_p' + gl]['r'] * press_mac_p
    for ii in range(ncond):
        u1 += corrs['corrs_k%d' % ii + gl]['u'] * phi_mac_s[..., [ii]]
        phi += corrs['corrs_k%d' % ii + gl]['r'] * phi_mac_p[..., [ii]]

    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            u1 += corrs['corrs_rs' + gl]['u_%d%d' % (ii, jj)]\
                * strain_mac_s[:, [kk]]
            phi += corrs['corrs_rs' + gl]['r_%d%d' % (ii, jj)]\
                * strain_mac_p[:, [kk]]

    u = displ_mac_s + eps0 * u1
    if map_flag:
        e_mac_Ys = [None] * macro['strain'].shape[1]
        for ii in range(dim):
            for jj in range(dim):
                kk = coor_to_sym(ii, jj, dim)
                mvar['svar'].set_data(macro['strain'][:, kk])
                mac_e_Ys = pb.evaluate('ev_volume_integrate.i2.Ys(svar)',
                                       mode='el_avg',
                                       var_dict={'svar': mvar['svar']})

                e_mac_Ys[kk] = mac_e_Ys.squeeze()

        e_mac_Ys = nm.vstack(e_mac_Ys).T[:, nm.newaxis, :, nm.newaxis]
    else:
        e_mac_Ys = macro['strain']

    mvar['r'].set_data(phi)
    E_mic = pb.evaluate('ev_grad.i2.Yp(r)', mode='el_avg',
                        var_dict={'r': mvar['r']}) / eps0

    mvar['u'].set_data(u1)
    e_mic = pb.evaluate('ev_cauchy_strain.i2.Ys(u)', mode='el_avg',
                        var_dict={'u': mvar['u']})

    e_mic += e_mac_Ys

    mvar['u'].set_data(u)
    uc = pb.evaluate('ev_integrate.i2.Ys(u)', mode='el_avg',
                        var_dict={'u': mvar['u']})

    # Stokes in Yf
    nvd = mvar['w'].field.n_vertex_dof
    nnod = corrs['corrs_psi' + gl]['p_0'].shape[0]  # only vertex dofs
    p1 = nm.zeros((nnod, 1), dtype=nm.float64)
    w = nm.zeros((nnod, dim), dtype=nm.float64)
    fluid = [v for v in pb.conf.materials.values() if v.name == 'fluid'][0]
    bar_eta = fluid.values['bar_eta']
    for key, val in corrs['corrs_psi' + gl].items():
        gp = press_mac_grad_f[:, [int(key[-1])]]
        if key[:2] == 'p_':
            p1 += -val * gp  # -/+ ???
        elif key[:2] == 'w_':
            w += -val[:nvd, :] * gp / bar_eta  # -/+ ???

    p = press_mac_f + eps0 * p1

    if map_flag:
        p_grad = press_mac_grad_f
    else:
        p_grad = p * 0 + press_mac_grad_f

    u_Ys = nm.zeros((mesh.n_nod, dim), dtype=nm.float64)
    u_Ys[Ys_map, :] = u.reshape(Ys_map.shape[0], dim)
    u_Yf = define_Yf_dvel(pb, u_Ys)

    mvar['uf'].set_data(w)
    grad_w = pb.evaluate('ev_grad.i2.Yf(uf)', mode='el_avg',
                         var_dict={'uf': mvar['uf']})
    wc = pb.evaluate('ev_integrate.i2.Yf(uf)', mode='el_avg',
                     var_dict={'uf': mvar['uf']})

    out = {
        'u': (u, 'u', 'p'),
        'uc': (uc, 'u', 'c'),
        'u1': (u1, 'u', 'p'),
        'e_mic': (e_mic, 'u', 'c'),
        'phi': (phi, 'r', 'p'),
        'E_mic': (E_mic, 'r', 'c'),
        'w': (w, 'w', 'p'),
        'grad_w': (grad_w.reshape((grad_w.shape[0], 1, 9, 1)), 'w', 'c'),
        'wc': (wc, 'w', 'c'),
        'uf_w': (u_Yf, 'w', 'p'),
        'p_grad': (p_grad, 'p', 'p'),
        'p': (p, 'p', 'p'),
        'p1': (p1, 'p', 'p'),
        'p_mac': (press_mac_f, 'p', 'p'),
        'pgrad_mac': (press_mac_grad_f, 'p', 'p'),
        'uf_p': (u_Yf, 'p', 'p'),
    }

    return data_to_struct(out)

def set_V_on_sf(ts, coor, problem=None, **kwargs):
    cmmap = problem.domain.regions['Gamma_sf'].vertices
    val = problem.dvelocity[cmmap]
    return val.flat


def define_Yf_dvel(pb, dvel):
    from sfepy.discrete import Problem

    dim = pb.domain.mesh.dim
    mx = nm.max(pb.domain.mesh.coors[pb.domain.regions['Yf'].vertices], axis=0)
    bbox = pb.domain.mesh.get_bounding_box()
    periodic = nm.abs(bbox[1] - mx) / (bbox[1] - bbox[0])
    epbcs, _ = get_periodic_bc([('uf', 'Yf')])
    for k in range(dim):
        if periodic[k] > 1e-3:
            del(epbcs['per_uf_' + 'xyz'[k]])

    conf = pb.conf.copy()
    conf.equations = {'eq': 'dw_lin_elastic.i2.Yf(fluid.D, vf, uf) = 0'}
    conf.edit('ebcs', {'fixed_uf': ('Gamma_sf', {'uf.all': 'set_V_on_sf'})})
    conf.edit('lcbcs', {})
    conf.edit('epbcs', epbcs)
    functions = {'set_V_on_sf': (set_V_on_sf,)}
    functions.update(conf._raw['functions'])
    conf.edit('functions', functions)
    aeps = nm.linalg.norm(dvel) * 1e-3
    conf.edit('solvers', {
        'ns': ('nls.newton', {'i_max': 1, 'eps_a': aeps,
                              'problem': 'nonlinear'}),
        'ls': ('ls.mumps', {})})
    lpb = Problem.from_conf(conf)
    lpb.dvelocity = dvel
    lpb.time_update()
    lpb.init_solvers()
    lpb.set_linear(False)
    dvel_f = lpb.solve().get_state_parts()

    nnod = pb.domain.regions['Yf'].vertices.shape[0]
    return dvel_f['uf'].reshape((nnod, dim))


class CorrDVel(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        pb = get_default(problem, self.problem)
        odir = pb.conf.options['output_dir']

        dim = self.dim
        domain = pb.domain
        mesh = domain.mesh
        mmap = domain.regions['Ys'].vertices
        cmap = domain.regions['Yf'].vertices

        lab = '_' + self.corr_flag
        if self.corr_flag == 'p':
            state_u = -data[self.requires[-1]].state['u']
        elif self.corr_flag[0] == 'r':
            state_u = data[self.requires[-1]].state['u']
        elif self.corr_flag == 'e':
            state_u = data[self.requires[-1]].states[self.idxs]['u']
            lab += '%d%d' % self.idxs

        out = nm.zeros((mesh.n_nod, dim), dtype=nm.float64)
        out[mmap, :] = state_u.reshape(mmap.shape[0], dim)
        out[cmap, :] = define_Yf_dvel(pb, out)

        if self.corr_flag == 'e':
            pis = data[self.requires[0]]
            out += pis.states[self.idxs]['V'].reshape(mesh.n_nod, dim)

        corr_sol = cb.CorrSolution(name=self.name + lab,
                                   state={'v': out.flatten()})

        mout = {'dvelocity': Struct(name='output_data', mode='vertex',
                                    data=out, dofs=None)}
        mesh.write(osp.join(odir, 'piezo_micro_dvelocity%s.vtk' % lab),
                   out=mout, io='auto')

        multiproc_dependecies['dvelocity' + lab] = out

        return corr_sol

def build_op_pi(val, ir, ic):
    pi = nm.zeros_like(val)
    pi[:, ir] = val[:, ic]
    pi.shape = (pi.shape[0] * pi.shape[1],)

    return pi


def create_pis(val, vname='u'):
    dim = val.shape[1]
    pis = nm.zeros((dim, dim), dtype=object)
    names = []
    for ir in range(dim):
        for ic in range(dim):
            pi = build_op_pi(val, ir, ic)
            pis[ir, ic] = {vname: pi}
            names.append('_%d%d' % (ir, ic))
    return names, pis


class CorrDVel_d_pis_u(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)
        domain = problem.domain
        mesh = domain.mesh

        mmap = domain.regions['Ys'].vertices
        key = list(data.keys())[0]
        dvel = data[key].state['v'].reshape((-1, mesh.dim))
        clist, dout = create_pis(dvel[mmap, :], vname='u')
        corr_sol = cb.CorrSolution(name=self.name,
                                   states=dout,
                                   components=clist)

        return corr_sol


def replace_dvel_in_def(coef, dvlab):
    def repl_in_list(svlist, key, dvlab):
        for k in range(len(svlist)):
            item = svlist[k]
            if isinstance(item, list):
                repl_in_list(item, key, dvlab)
            elif item == key:
                svlist[k] = (item[0], item[1] + dvlab, item[2])

    out = deepcopy(coef)
    # replace "dvelocity" by "dvelocity_xy"
    req = out['requires']
    if 'dvelocity' in req:
        idx = req.index('dvelocity')
        req[idx] = 'dvelocity' + dvlab

    if 'd_pis_u' in req:
        idx = req.index('d_pis_u')
        req[idx] = 'd_pis_u' + dvlab

    if 'set_variables' in out:
        repl_in_list(out['set_variables'], ('V', 'dvelocity', 'v'), dvlab)
        repl_in_list(out['set_variables'], ('U1', 'd_pis_u', 'u'), dvlab)
        repl_in_list(out['set_variables'], ('U2', 'd_pis_u', 'u'), dvlab)

    if out['class'] is cb.CoefEval:
        repl = []
        new_req = []

        for jj in out['requires']:
            if jj[0:3] == 'c.s' or jj == 'c.divV_Y0' or jj == 'c.divV_Yf':
                new_req.append(jj + dvlab)
                repl.append((jj, jj + dvlab))
            else:
                new_req.append(jj)

        out['requires'] = new_req
        for jj in repl:
            out['expression'] = out['expression'].replace(jj[0], jj[1])

    return out


def define(eps0=1e-3,
           filename_mesh=osp.join(wdir, 'meshes', 'mesh_micro_1D.vtk'),
           mat_mode='elastic_part', flag='',
           filename_coefs=None,
           output_dir='output'):

    if filename_coefs is None:
        filename_coefs = f'coefs_poropiezo{flag}'

    filename_mesh = osp.join(wdir, filename_mesh)
    mesh = Mesh.from_file(filename_mesh)
    dim = mesh.dim
    n_mat_woc = 2 if mat_mode is None else 3
    n_conduct = len(nm.unique(mesh.cmesh.cell_groups)) - n_mat_woc

    print('micromode:')
    print('  num. of conductors - %d' % n_conduct)
    print('  elastic part - %s' % (mat_mode == 'elastic_part'))

    sym_eye = 'nm.array([1, 1, 0])' if dim == 2\
        else 'nm.array([1, 1, 1, 0, 0, 0])'

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(dim, bbox[0], bbox[1], eps=1e-3)

    regions.update({
        'Y': 'all',
        'Gamma_sf': ('r.Ys *s r.Yf', 'facet', 'Ys'),
        # channel / inclusion
        'Yf0': ('r.Yf -s r.Gamma_sf', 'facet'),
    })

    regions.update(get_periodic_regions('Ys'))

    if mat_mode is None:
        # parts: piezo, fluid
        regions.update({
            'Ym': 'cells of group 1',
            'Yf': 'cells of group 2',
            'Yp': ('copy r.Ym', 'cell'),
            'Ye': ('copy r.Ym', 'cell'),
        })
        regions.update(get_periodic_regions('Yp'))
    else:
        # parts: piezo, elastic, fluid
        regions.update({
            'Ye': 'cells of group 1',
            'Yp': 'cells of group 2',
            'Yf': 'cells of group 3',
            'Ym': ('r.Ye +v r.Yp', 'cell'),
        })
        regions.update(get_periodic_regions('Yp'))

    if n_conduct > 0:
        regions.update({
            'Yc': (' +v '.join(['r.Yc%d' % k for k in range(n_conduct)]),
                   'cell'),
            'Ys': ('r.Ym +v r.Yc', 'cell'),
            'Gamma_mc': ('r.Ym *s r.Yc', 'facet', 'Ym'),
        })
        for k in range(n_conduct):
            sk = '%d' % k
            regions.update({
                'Yc' + sk: 'cells of group %d' % (n_mat_woc + 1 + k),
                'Gamma_c' + sk: ('r.Ym *s r.Yc' + sk, 'facet', 'Ym'),
            })

    else:
        regions.update({'Ys': ('copy r.Ym', 'cell')})


    mat_id = 2 if mat_mode is None else 3
    regions.update(get_periodic_regions('Yf0', label='Yf',
                                        mesh_data=(mat_id, mesh)))

    options = {
        'coefs_filename': filename_coefs,
        'volume': {
            'variables': ['svar'],
            'expression': 'd_volume.i2.Y(svar)',
        },
        'coefs': 'coefs',
        'requirements': 'requirements',
        'ls': 'ls',
        'file_per_var': True,
        'absolute_mesh_path': True,
        'multiprocessing': False,
        # 'multiprocessing': True,
        'output_dir': output_dir,
        'return_all': True,
        'recovery_hook': recovery_micro,
    }

    fields = {
        'displacement': ('real', 'vector', 'Ys', 1),
        'potential': ('real', 'scalar', 'Yp', 1),
        'sfield': ('real', 'scalar', 'Y', 1),
        'velocity': ('real', 'vector', 'Yf', 2),
        'pressure': ('real', 'scalar', 'Yf', 1),
        'dvelocity': ('real', 'vector', 'Y', 1),
        'displacement_Yf': ('real', 'vector', 'Yf', 1),
    }

    variables = {
        # displacement
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'Pi_u': ('parameter field', 'displacement', 'u'),
        'U1': ('parameter field', 'displacement', '(set-to-None)'),
        'U2': ('parameter field', 'displacement', '(set-to-None)'),
        # electric potential
        'r': ('unknown field', 'potential'),
        's': ('test field', 'potential', 'r'),
        'Pi_r': ('parameter field', 'potential', 'r'),
        'R1': ('parameter field', 'potential', '(set-to-None)'),
        'R2': ('parameter field', 'potential', '(set-to-None)'),
        'svar': ('parameter field', 'sfield', '(set-to-None)'),
        # fluid pressure
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'P1': ('parameter field', 'pressure', '(set-to-None)'),
        'P2': ('parameter field', 'pressure', '(set-to-None)'),
        'ls': ('unknown field', 'pressure'),
        'lv': ('test field', 'pressure', 'ls'),
        # fluid velocity
        'w': ('unknown field', 'velocity'),
        'z': ('test field', 'velocity', 'w'),
        'Pi_w': ('parameter field', 'velocity', 'w'),
        'W1': ('parameter field', 'velocity', '(set-to-None)'),
        'W2': ('parameter field', 'velocity', '(set-to-None)'),
        # design velocity
        'V': ('parameter field', 'dvelocity', '(set-to-None)'),
        'uf': ('unknown field', 'displacement_Yf'),
        'vf': ('test field', 'displacement_Yf', 'uf'),
    }

    epbcs, periodic = get_periodic_bc([('u', 'Ys'), ('r', 'Yp'), ('w', 'Yf')],
                                      regions=regions)

    mat_g_sc, mat_d_sc = eps0, eps0**2

    materials = {
        'matrix': ({
            'D': {'Yp': D_piezo},
        },),
        'piezo': ({
            'g': g_piezo / mat_g_sc,
            'd': d_piezo / mat_d_sc,
        },),
        'fluid': ({  # water
            'one': 1.0,
            'gamma': 1.0 / 2.15e9,
            'bar_eta': 8.9e-4 / eps0**2,  # dynamic viscosity
            'D': stiffness_from_youngpoisson(dim, 1.0, 0.3),
        },),
    }

    if mat_mode is not None:
        materials['matrix'][0]['D'].update({'Ye': D_elast})

    functions = {
        'match_x_plane': (match_x_plane,),
        'match_y_plane': (match_y_plane,),
        'match_z_plane': (match_z_plane,),
    }

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
        'fixed_r': ('Gamma_mc', {'r.all': 0.0}),
        'fixed_w': ('Gamma_sf', {'w.all': 0.0}),
    }

    fixed_r = ['fixed_r']

    integrals = {
        'i2': 2,
        'i3': 3,
    }

    solvers = {
        'ls': ('ls.mumps', {}),
        'ns_em6': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-6,
                   'eps_r': 1e-3,
                   'problem': 'nonlinear'}),
        'ns_em12': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-12,
                   'eps_r': 1e-3,
                   'problem': 'nonlinear'}),
        'ns_em1': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-1,
                   'eps_r': 1e-3,
                   'problem': 'nonlinear'}),
    }

    coefs = {
        'A': {
            'requires': ['pis_u', 'corrs_rs'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' + dw_diffusion.i2.Yp(piezo.d, R1, R2)',
            'set_variables': [[('U1', ('corrs_rs', 'pis_u'), 'u'),
                               ('R1', 'corrs_rs', 'r')],
                              [('U2', ('corrs_rs', 'pis_u'), 'u'),
                               ('R2', 'corrs_rs', 'r')]],
            'class': cb.CoefSymSym,
        },
        'B1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R2)',
            'set_variables': [('U1', 'pis_u', 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R2', 'corrs_p', 'r')],
            'class': cb.CoefSym,
        },
        'B': {
            'requires': ['c.Phi', 'c.B1'],
            'expression': 'c.B1 + c.Phi * %s' % sym_eye,
            'class': cb.CoefEval,
        },
        'N': {
            'status': 'auxiliary',
            'requires': ['corrs_p'],
            'expression': 'dw_surface_ltr.i2.Gamma_sf(U1)',
            'set_variables': [('U1', 'corrs_p', 'u')],
            'class': cb.CoefOne,
        },
        'M': {
            'requires': ['c.Phi', 'c.N'],
            'expression': 'c.N + c.Phi * %e' % materials['fluid'][0]['gamma'],
            'class': cb.CoefEval,
        },
        'K': {
            'requires': ['corrs_psi'],
            'expression': 'dw_div_grad.i3.Yf(W1, W2)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                                ('W2', 'corrs_psi', 'w')],
            'class': cb.CoefDimDim,
        },
        'Phi': {
            'requires': ['c.vol'],
            'expression': 'c.vol["fraction_Yf"]',
            'class': cb.CoefEval,
        },
        'vol': {
            'regions': ['Ym', 'Yf'] + ['Yc%d' % k for k in range(n_conduct)],
            'expression': 'd_volume.i2.%s(svar)',
            'class': cb.VolumeFractions,
        },
        'eps0': {
            'requires': [],
            'expression': '%e' % eps0,
            'class': cb.CoefEval,
        },
        'bar_eta': {
            'expression': '%e' % materials['fluid'][0]['bar_eta'],
            'class': cb.CoefEval,
        },
        'filenames': {},
        'divV_Y0': {
            'requires': ['dvelocity'],
            'expression': 'ev_div.i2.Y(V)',
            'set_variables': [('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'divV_Yf': {
            'requires': ['dvelocity'],
            'expression': 'ev_div.i2.Yf(V)',
            'set_variables': [('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sPhi': {
            'status': 'auxiliary',
            'requires': ['c.divV_Y0', 'c.divV_Yf', 'c.vol'],
            'expression': 'c.divV_Yf - c.vol["fraction_Yf"] * c.divV_Y0',
            'class': cb.CoefEval,
        },
        ##### sA
        'sA1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'dvelocity'],
            'expression': '   ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)'
                          ' - ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)',
            'set_variables': [[('U1', ('corrs_rs', 'pis_u'), 'u'),
                               ('R1', 'corrs_rs', 'r')],
                              [('U2', ('corrs_rs', 'pis_u'), 'u'),
                               ('R2', 'corrs_rs', 'r')],
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSymSym,
        },
        'sA2': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'd_pis_u', 'dvelocity'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R2)'
                          ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R1, V)',
            'set_variables': [[('U1', 'd_pis_u', 'u'),
                               ('R1', 'corrs_rs', 'r')],
                              [('U2', ('corrs_rs', 'pis_u'), 'u'),
                               ('R2', 'corrs_rs', 'r')],
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSymSym,
        },
        'sA': {
            'requires': ['c.sA1', 'c.sA2'],
            'expression': 'c.sA1 + c.sA2 + c.sA2.T',
            'class': cb.CoefEval,
        },
        #### sB
        'sB1_div1': {
            'status': 'auxiliary',
            'requires': ['corr_one', 'corrs_rs', 'dvelocity'],
            'expression': 'ev_sd_div.i2.Ys(U1, svar, V)',
            'set_variables': [('U1', 'corrs_rs', 'u'),
                              ('svar', 'corr_one', 'sv'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSym,
        },
        'sB1_div2': {
            'status': 'auxiliary',
            'requires': ['corrs_rs', 'dvelocity'],
            'expression': 'ev_div.i2.Ys(U1)',
            'set_variables': [('U1', 'corrs_rs', 'u')],
            'class': cb.CoefSym,
        },
        'sB2': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'd_pis_u'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R1)',
            'set_variables': [('U1', 'd_pis_u', 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R1', 'corrs_p', 'r')],
            'class': cb.CoefSym,
        },
        'sB3': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'corrs_p', 'dvelocity'],
            'expression': '   ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)'
                          ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R1, V)'
                          ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R2, V)'
                          ' - ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)',
            'set_variables': [('U1', ('corrs_rs', 'pis_u'), 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R1', 'corrs_rs', 'r'),
                              ('R2', 'corrs_p', 'r'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSym,
        },
        'sB': {
            'requires': ['c.sB1_div1', 'c.sB1_div2', 'c.sB2', 'c.sB3',
                         'c.divV_Y0', 'c.sPhi'],
            'expression': f'{sym_eye} * c.sPhi' +
                          ' - (c.sB1_div1 - c.divV_Y0 * c.sB1_div2)'
                          ' + c.sB2 + c.sB3',
            'class': cb.CoefEval,
        },
        #### sM
        'sMsurf1': {
            'status': 'auxiliary',
            'requires': ['corr_one', 'corrs_p', 'dvelocity'],
            'expression': 'ev_sd_div.i2.Ys(U1, svar, V)',
            'set_variables': [('U1', 'corrs_p', 'u'),
                              ('svar', 'corr_one', 'sv'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sMsurf2': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'dvelocity'],
            'expression': 'ev_div.i2.Ys(U1)',
            'set_variables': [('U1', 'corrs_p', 'u')],
            'class': cb.CoefOne,
        },
        'sMsurf': {
            'status': 'auxiliary',
            'requires': ['c.sMsurf1', 'c.sMsurf2', 'c.divV_Y0'],
            # \int_{Gamma_c} x \cdot n = - \int_{Ys} div(z) + ...
            'expression': '-(c.sMsurf1 - c.divV_Y0 * c.sMsurf2)',
            'class': cb.CoefEval,
        },
        'sM2': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'dvelocity'],
            'expression': ' 2*ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R1, V)'
                          ' + ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)'
                          ' - ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)',
            'set_variables': [('U1', 'corrs_p', 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R1', 'corrs_p', 'r'),
                              ('R2', 'corrs_p', 'r'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sM': {
            'requires': ['c.sMsurf', 'c.sM2', 'c.sPhi'],
            'expression': '%e * c.sPhi' % materials['fluid'][0]['gamma'] +
                          ' - 2*c.sMsurf + c.sM2',
            'class': cb.CoefEval,
        },
        ### sK
        'sK2': {
            'status': 'auxiliary',
            'requires': ['corrs_psi'],
            'expression': 'dw_stokes.i3.Yf(W1, P1)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                              ('P1', 'corrs_psi', 'p')],
            'class': cb.CoefDimDim,
        },
        'sK3': {
            'status': 'auxiliary',
            'requires': ['pis_w', 'corrs_psi', 'dvelocity'],
            'expression': '   ev_sd_volume_dot.i3.Yf(W1, W2, V)'
                          ' + ev_sd_div.i3.Yf(W1, P2, V)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                              [('W2', 'pis_w', 'w'),
                               ('P2', 'corrs_psi', 'p')],
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefDimDim,
        },
        'sK4': {
            'status': 'auxiliary',
            'requires': ['corrs_psi', 'dvelocity'],
            'expression': 'ev_sd_div_grad.i3.Yf(W1, W2, V)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                              ('W2', 'corrs_psi', 'w'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefDimDim,
        },
        'sK': {
            'requires': ['c.K', 'c.sK2', 'c.sK3', 'c.sK4', 'c.divV_Y0'],
            'expression': '-(c.K + c.sK2 + c.sK2.T) * c.divV_Y0'
                          ' - c.sK4 + c.sK3 + c.sK3.T',
            'class': cb.CoefEval,
        },
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'pis_r': {
            'variables': ['r'],
            'class': cb.ShapeDim,
        },
        'corrs_rs': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u'] + fixed_r,
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                     - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                   = - dw_lin_elastic.i2.Ys(matrix.D, v, Pi_u)""",
                'eq2':
                    """
                     - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                     - dw_diffusion.i2.Yp(piezo.d, s, r)
                     = dw_piezo_coupling.i2.Yp(piezo.g, Pi_u, s)""",
            },
            'set_variables': [('Pi_u', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_rs' + flag,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em4'},
        },
        'corrs_p': {
            'requires': [],
            'ebcs': ['fixed_u'] + fixed_r,
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                     - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                     = dw_surface_ltr.i2.Gamma_sf(v)""",
                'eq2':
                    """
                     - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                     - dw_diffusion.i2.Yp(piezo.d, s, r)
                     = 0"""
            },
            'class': cb.CorrOne,
            'save_name': 'corrs_p' + flag,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
        },
        'corrs_rho': {
            'requires': [],
            'ebcs': ['fixed_u', 'fixed_r'],
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                        - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                        = 0""",
                'eq2':
                    """
                        - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                        - dw_diffusion.i2.Yp(piezo.d, s, r)
                        =
                        - dw_surface_integrate.i2.Gamma_sf(s)"""
                },
            'class': cb.CorrOne,
            'save_name': 'corrs_p' + flag,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
        },
        'pis_w': {
            'variables': ['w'],
            'class': cb.OnesDim,
        },
        'corrs_psi': {
            'requires': ['pis_w'],
            'ebcs': ['fixed_w'],
            'epbcs': periodic['per_w'],
            'lcbcs': ['imv'],
            'is_linear': False,
            'equations': {
                'balance_of_forces':
                    """dw_div_grad.i3.Yf(z, w)
                        - dw_stokes.i3.Yf(z, p)
                        =
                        dw_volume_dot.i3.Yf(z, Pi_w)""",
                'incompressibility':
                    """
                        - dw_stokes.i3.Yf(w, q)
                        + dw_dot.i3.Yf(q, ls)
                        = 0""",
                'imv': 'dw_dot.i3.Yf(lv, p) = 0',
            },
            'set_variables': [('Pi_w', 'pis_w', 'w')],
            'class': cb.CorrDim,
            'save_name': 'corrs_psi' + flag,
            'dump_variables': ['w', 'p'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em12'},
        },
        #  = V_j \delta_{ik} in Ys
        'd_pis_u': {
            'requires': ['dvelocity'],
            'class': CorrDVel_d_pis_u,
        },
        'pis_v': {
            'variables': ['V'],
            'class': cb.ShapeDimDim,
        },
        'corr_one': {
            'variable': 'sv',
            'expression': "nm.ones((problem.fields['sfield'].n_vertex_dof, 1), dtype=nm.float64)",
            'class': cb.CorrEval,
        },
        'dvelocity': {
            'variable': 'v',
            'expression': 'problem.conf.inter_data',
            'class': cb.CorrEval,
            # 'save_name': 'design_velocity',
            # 'dump_variables': ['v'],
        },
        'dvelocity_p': {
            'requires': ['corrs_p'],
            'dim': dim,
            'corr_flag': 'p',
            'class': CorrDVel,
            # 'save_name': 'dvelocity_p',
            # 'dump_variables': ['v'],
        },
    }

    for k in range(n_conduct):
        sk = '%d' % k

        materials['matrix'][0]['D'].update({'Yc' + sk: D_cond})

        ebcs.update({
            'fixed_r1_k_' + sk: ('Gamma_c' + sk, {'r.0': 1.0}),
            'fixed_r0_k_' + sk: ('Gamma_c' + sk, {'r.0': 0.0}),
        })

        fixed_r0_k = ['fixed_r0_k_%d' % ii for ii in range(n_conduct)
                        if not ii == k]

        requirements.update({
            'corrs_k' + sk: {
                'requires': ['pis_r'],
                'ebcs': ['fixed_u', 'fixed_r1_k_' + sk] + fixed_r0_k,
                'epbcs': periodic['per_u'] + periodic['per_r'],
                'is_linear': True,
                'equations': {
                    'eq1':
                        """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                        - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                        = 0""",
                    'eq2':
                        """
                        - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                        - dw_diffusion.i2.Yp(piezo.d, s, r)
                        = 0"""
                    },
                'class': cb.CorrOne,
                'save_name': 'corrs_k' + sk + flag,
                'dump_variables': ['u', 'r'],
                'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
            },
        })

        coefs.update({
            'V' + sk: {
                'requires': ['pis_u', 'corrs_k' + sk],
                'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                                ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R2)',
                'set_variables': [('U1', 'pis_u', 'u'),
                                    ('U2', 'corrs_k' + sk, 'u'),
                                    ('R2', 'corrs_k' + sk, 'r')],
                'class': cb.CoefSym,
            },
            'Z' + sk: {
                'requires': ['corrs_k' + sk],
                'expression': 'dw_surface_ltr.i2.Gamma_sf(U1)',
                'set_variables': [('U1', 'corrs_k' + sk, 'u')],
                'class': cb.CoefOne,
            },
        })


    lcbcs = {
        'imv': ('Yf', {'ls.all': None}, None, 'integral_mean_value'),
    }

    dvlist = ['_p']

    for k in range(n_conduct):
        sk = '%d' % k
        ck = '_' + sk
        coefs.update({
            'sV1' + ck: {
                'status': 'auxiliary',
                'requires': ['d_pis_u', 'corrs_k' + sk],
                'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                              ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R1)',
                'set_variables': [('U1', 'd_pis_u', 'u'),
                                  ('U2', 'corrs_k' + sk, 'u'),
                                  ('R1', 'corrs_k' + sk, 'r')],
                'class': cb.CoefSym,
            },
            'sV2' + ck: {
                'status': 'auxiliary',
                'requires': ['pis_u', 'corrs_rs', 'corrs_k' + sk, 'dvelocity'],
                'expression': '   ev_sd_lin_elastic.i2.Ys(matrix.D, U2, U1, V)'
                              ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R1, V)'
                              ' - ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)'
                              ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R2, V)',
                'set_variables': [('R1', 'corrs_rs', 'r'),
                                  ('R2', 'corrs_k' + sk, 'r'),
                                  ('U1', 'corrs_k' + sk, 'u'),
                                  ('U2', ('corrs_rs', 'pis_u'), 'u'),
                                  ('V', 'dvelocity', 'v')],
                'class': cb.CoefSym,
            },
            'sV' + sk: {
                'requires': ['c.sV1' + ck, 'c.sV2' + ck],
                'expression': 'c.sV1%s + c.sV2%s' % (ck, ck),
                'class': cb.CoefEval,
            },
            #### sZ
            'sZsurf1' + ck: {
                'status': 'auxiliary',
                'requires': ['corr_one', 'corrs_k' + sk, 'dvelocity'],
                'expression': 'ev_sd_div.i2.Ys(U1, svar, V)',
                'set_variables': [('U1', 'corrs_k' + sk, 'u'),
                                  ('svar', 'corr_one', 'sv'),
                                  ('V', 'dvelocity', 'v')],
                'class': cb.CoefOne,
            },
            'sZsurf2' + ck: {
                'status': 'auxiliary',
                'requires': ['corrs_k' + sk, 'dvelocity'],
                'expression': 'ev_div.i2.Ys(U1)',
                'set_variables': [('U1', 'corrs_k' + sk, 'u')],
                'class': cb.CoefOne,
            },
            'sZsurf' + ck: {
                'status': 'auxiliary',
                'requires': ['c.sZsurf1' + ck, 'c.sZsurf2' + ck, 'c.divV_Y0'],
                # \int_{Gamma_c} z \cdot n = - \int_{Ys} div(z) + ...
                'expression': '-(c.sZsurf1%s - c.divV_Y0 * c.sZsurf2%s)' % (ck, ck),
                'class': cb.CoefEval,
            },  
            'sZ1' + ck: {
                'status': 'auxiliary',
                'requires': ['corrs_p', 'corrs_k' + sk, 'dvelocity'],
                'expression': '   ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R1, V)'
                              ' - ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)'
                              ' + ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R2, V)'
                              ' + ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)',
                'set_variables': [('U1', 'corrs_p', 'u'),
                                  ('U2', 'corrs_k' + sk, 'u'),
                                  ('R1', 'corrs_k' + sk, 'r'),
                                  ('R2', 'corrs_p', 'r'),
                                  ('V', 'dvelocity', 'v')],
                'class': cb.CoefOne,
            },
            'sZ' + sk: {
                'requires': ['c.sZ1' + ck, 'c.sZsurf' + ck],
                'expression': 'c.sZ1%s - c.sZsurf%s' % (ck, ck),
                'class': cb.CoefEval,
            },
        })

        dvlab = '_r' + sk
        dvlist.append(dvlab)
        requirements.update({
            'dvelocity' + dvlab: {
                'requires': ['corrs_k' + sk],
                'dim': dim,
                'corr_flag': 'r' + sk,
                'class': CorrDVel,
            },
        })

    for ii, irc in enumerate(iter_sym(dim)):
        dvlab = '_e%d%d' % irc
        dvlist.append(dvlab)
        requirements.update({
            'dvelocity' + dvlab: {
                'requires': ['pis_v', 'corrs_rs'],
                'dim': dim,
                'corr_flag': 'e',
                'idxs': irc,
                'class': CorrDVel,
            },
        })

    coefs_keys = [k for k in coefs.keys()
                  if (k.startswith('s') or k.startswith('divV'))]

    for dvlab in dvlist:
        requirements.update({
            'd_pis_u' + dvlab: {
                'requires': ['dvelocity' + dvlab],
                'class': CorrDVel_d_pis_u,
            },
        })

        for ck in coefs_keys:
            coefs[ck + dvlab] = replace_dvel_in_def(coefs[ck], dvlab)

    for ck in coefs_keys:
        del(coefs[ck])

    with open(osp.join(options['output_dir'], 'coefs_poropiezo_def.py'), 'wt') as f:
        for ck in coefs.keys():
            f.write('===== %s =====\n' % ck)
            cv = coefs[ck]
            for k in cv.keys():
                f.write('    %s: %s\n' % (k, str(cv[k])))

    return locals()
