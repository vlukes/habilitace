import os.path as osp
import numpy as nm
from copy import deepcopy
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.homogenization.utils import coor_to_sym, define_box_regions, iter_sym
import sfepy.homogenization.coefs_base as cb
from sfepy.discrete.fem.mesh import Mesh
from sfepy.discrete.fem.periodic import match_grid_plane
from sfepy.base.base import Struct, get_default
import sfepy.base.multiproc as multiproc

wdir = osp.dirname(__file__)

mp_module, _ = multiproc.get_multiproc()
multiproc_dependecies = mp_module.get_dict('dependecies', clear=True)


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


def data_to_struct(data):
    out = {}
    for k, v in data.items():
        out[k] = Struct(name='output_data',
                        mode='cell' if v[2] == 'c' else 'vertex',
                        data=v[0],
                        var_name=v[1],
                        dofs=None)

    return out


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


def define(eps0=1e-4,
           young=3e9, poisson=0.34, eta=0.950, gamma=1.0/4.35e9,
           filename_mesh=osp.join(wdir, 'meshes', 'matrix_fluid_c2r1.vtk'),
           filename_coefs='coefs',
           multiprocessing=False,
           output_dir='output'
           ):

    options = {
        'coefs_filename': filename_coefs,
        'volume': {
            'variables': ['svar'],
            'expression': 'd_volume.i2.Y(svar)',
        },
        'coefs': 'coefs',
        'requirements': 'requirements',
        'output_dir': output_dir,
        'file_per_var': True,
        'absolute_mesh_path': True,
        'multiprocessing': multiprocessing,
        'return_all': True,
        # 'recovery_hook': recovery_micro,
    }
    

    mesh = Mesh.from_file(filename_mesh)
    dim = mesh.dim

    sym_eye = f'nm.array({"[1, 1, 0]" if dim == 2 else "[1, 1, 1, 0, 0, 0]"})'

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(dim, bbox[0], bbox[1], eps=1e-3)

    regions.update({
        'Y': 'all',
        'Ys': 'cells of group 1',
        'Yf': 'cells of group 2',
        'Gamma_sf': ('r.Ys *s r.Yf', 'facet', 'Ys'),
        'Yf0': ('r.Yf -s r.Gamma_sf', 'facet'),
    })

    regions.update(get_periodic_regions('Ys'))
    regions.update(get_periodic_regions('Yf0', label='Yf', mesh_data=(2, mesh)))

    fields = {
        'sfield': ('real', 'scalar', 'Y', 1),
        'displacement': ('real', 'vector', 'Ys', 1),
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
        'svar': ('parameter field', 'sfield', '(set-to-None)'),
        # fluid pressure
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'P1': ('parameter field', 'pressure', '(set-to-None)'),
        'P2': ('parameter field', 'pressure', '(set-to-None)'),
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

    epbcs, periodic = get_periodic_bc([('u', 'Ys'), ('w', 'Yf'), ('p', 'Yf')], regions=regions)

    c43, c23 = 4./3, 2./3
    bar_eta = eta / (eps0**2)

    materials = {
        'solid': ({
            'D': stiffness_from_youngpoisson(3, young, poisson),
        },),
        'fluid': ({  # water
            'gamma': gamma,
            # 'bar_eta': 8.9e-4 / (eps0**2),  # dynamic viscosity
            'bar_eta': bar_eta,  # dynamic viscosity
            'D': nm.array([[c43, -c23, -c23, 0, 0, 0],
                           [-c23, c43, -c23, 0, 0, 0],
                           [-c23, -c23, c43, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0], 
                           [0, 0, 0, 0, 1, 0], 
                           [0, 0, 0, 0, 0, 1]]) * eta,
        },),
    }

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
        'fixed_w': ('Gamma_sf', {'w.all': 0.0}),
    }

    functions = {
        'match_x_plane': (match_x_plane,),
        'match_y_plane': (match_y_plane,),
        'match_z_plane': (match_z_plane,),
    }

    integrals = {
        'i2': 2,
        'i3': 3,
    }

    solvers = {
        'ls': ('ls.mumps', {}),
        'ns': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-6,
            'eps_r': 1e-3,
            'problem': 'nonlinear'}),
    }

    coefs = {
        'A': {
            'requires': ['pis_u', 'corrs_rs'],
            'expression': 'dw_lin_elastic.i2.Ys(solid.D, U1, U2)',
            'set_variables': [('U1', ('corrs_rs', 'pis_u'), 'u'),
                              ('U2', ('corrs_rs', 'pis_u'), 'u')],
            'class': cb.CoefSymSym,
        },
        'vol': {
            'regions': ['Ys', 'Yf'],
            'expression': 'd_volume.i2.%s(svar)',
            'class': cb.VolumeFractions,
        },
        'eps0': {
            'requires': [],
            'expression': f'{eps0}',
            'class': cb.CoefEval,
        },
        'filenames': {},
        'B1': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'pis_u'],
            'expression': 'dw_lin_elastic.i2.Ys(solid.D, U1, U2)',
            'set_variables': [('U1', 'pis_u', 'u'),
                              ('U2', 'corrs_p', 'u')],
            'class': cb.CoefSym,
        },
        'B': {
            'requires': ['c.Phi', 'c.B1'],
            'expression': f'c.B1 + c.Phi * {sym_eye}',
            'class': cb.CoefEval,
        },
        'N': {
            'status': 'auxiliary',
            'requires': ['corrs_p'],
            'expression': 'dw_lin_elastic.i2.Ys(solid.D, U1, U1)',
            'set_variables': [('U1', 'corrs_p', 'u')],
            'class': cb.CoefOne,
        },
        'M': {
            'requires': ['c.Phi', 'c.N'],
            'expression': f"c.N + c.Phi * {materials['fluid'][0]['gamma']}",
            'class': cb.CoefEval,
        },
        'K': {
            'requires': ['corrs_psi'],
            'expression': 'dw_lin_elastic.i3.Yf(fluid.D, W1, W2)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                              ('W2', 'corrs_psi', 'w')],
            'class': cb.CoefDimDim,
        },
        'Phi': {
            'requires': ['c.vol'],
            'expression': 'c.vol["fraction_Yf"]',
            'class': cb.CoefEval,
        },
        'bar_eta': {
            'expression': f"{materials['fluid'][0]['bar_eta']}",
            'class': cb.CoefEval,
        },
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'corrs_rs': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u'],
            'epbcs': periodic['per_u'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Ys(solid.D, v, u)
                   = - dw_lin_elastic.i2.Ys(solid.D, v, Pi_u)""",
            },
            'set_variables': [('Pi_u', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_rs',
            'solvers': {'ls': 'ls', 'nls': 'ns'},
        },
        'corrs_p': {
            'requires': [],
            'ebcs': ['fixed_u'],
            'epbcs': periodic['per_u'],
            'is_linear': True,
            'equations': {
                'balance_of_forces':
                    """dw_lin_elastic.i2.Ys(solid.D, v, u)
                     = dw_surface_ltr.i2.Gamma_sf(v)""",
            },
            'class': cb.CorrOne,
            'save_name': 'corrs_p',
            'solvers': {'ls': 'ls', 'nls': 'ns'},
        },
        'pis_w': {
            'variables': ['w'],
            'class': cb.OnesDim,
        },
        'corrs_psi': {
            'requires': ['pis_w'],
            'ebcs': ['fixed_w'],
            'epbcs': periodic['per_w'] + periodic['per_p'],
            'is_linear': False,
            'equations': {
                'balance_of_forces':
                    """   dw_lin_elastic.i3.Yf(fluid.D, z, w)
                        + dw_v_dot_grad_s.i3.Yf(z, p)
                        =
                          dw_volume_dot.i3.Yf(z, Pi_w)""",
                'incompressibility':
                    """   dw_v_dot_grad_s.i3.Yf(w, q)
                        + dw_dot.i3.Yf(fluid.gamma, q, p)
                        = 0""",
            },
            'set_variables': [('Pi_w', 'pis_w', 'w')],
            'class': cb.CorrDim,
            'save_name': 'corrs_psi',
            'solvers': {'ls': 'ls', 'nls': 'ns_flow'},
        },
    }

    coefs.update({
        'divV_Y0': {
            'requires': ['dvelocity'],
            'expression': 'ev_div.i2.Y(V)',
            'set_variables': [('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        ##### sA
        'sA1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'dvelocity'],
            'expression': 'ev_sd_lin_elastic.i2.Ys(solid.D, U1, U2, V)',
            'set_variables': [('U1', ('corrs_rs', 'pis_u'), 'u'),
                              ('U2', ('corrs_rs', 'pis_u'), 'u'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSymSym,
        },
        'sA': {
            'requires': ['c.A', 'c.sA1', 'c.divV_Y0'],
            'expression': 'c.sA1 - c.A * c.divV_Y0',
            'class': cb.CoefEval,
        },
        'divV_Yf': {
            'requires': ['dvelocity'],
            'expression': 'ev_div.i2.Yf(V)',
            'set_variables': [('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sPhi': {
            'requires': ['c.divV_Yf', 'c.divV_Y0', 'c.Phi'],
            'expression': 'c.divV_Yf - c.Phi * c.divV_Y0',
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
            'expression': 'dw_lin_elastic.i2.Ys(solid.D, U1, U2)',
            'set_variables': [('U1', 'd_pis_u', 'u'),
                              ('U2', 'corrs_p', 'u')],
            'class': cb.CoefSym,
        },
        'sB3': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'corrs_p', 'dvelocity'],
            'expression': 'ev_sd_lin_elastic.i2.Ys(solid.D, U1, U2, V)',
            'set_variables': [('U1', ('corrs_rs', 'pis_u'), 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSym,
        },
        'sB': {
            'requires': ['c.sB1_div1', 'c.sB1_div2', 'c.sB2', 'c.sB3',
                         'c.divV_Y0', 'c.sPhi'],
            'expression': f'c.sPhi * {sym_eye}' +
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
            # \int_{Gamma_f} x \cdot n = - \int_{Ys} div(z) + ...
            'expression': '-(c.sMsurf1 - c.divV_Y0 * c.sMsurf2)',
            'class': cb.CoefEval,
        },
        'sM1': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'dvelocity'],
            'expression': 'ev_sd_lin_elastic.i2.Ys(solid.D, U1, U1, V)',
            'set_variables': [('U1', 'corrs_p', 'u'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sM': {
            'requires': ['c.sMsurf', 'c.N', 'c.sM1', 'c.sPhi', 'c.divV_Y0'],
            'expression': f"c.sPhi * {materials['fluid'][0]['gamma']}" +
                          ' - 2*c.sMsurf - c.sM1 + c.N * c.divV_Y0',
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
            'expression': '   ev_sd_dot.i3.Yf(W1, W2, V)'
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
    })

    dvlist = []

    for irc in iter_sym(dim):
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

    requirements.update({
        'pis_v': {
            'variables': ['V'],
            'class': cb.ShapeDimDim,
        },
        'corr_one': {
            'variable': 'sv',
            'expression': "nm.ones((problem.fields['sfield'].n_vertex_dof, 1), dtype=nm.float64)",
            'class': cb.CorrEval,
        },
        'dvelocity_p': {
            'requires': ['corrs_p'],
            'dim': dim,
            'corr_flag': 'p',
            'class': CorrDVel,
        },
    })

    dvlist.append('_p')

    for dvlab in dvlist:
        requirements.update({
            #  = V_j \delta_{ik} in Ys
            'd_pis_u' + dvlab: {
                'requires': ['dvelocity' + dvlab],
                'variables': ['u'],
                'class': CorrDVel_d_pis_u,
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
    
    return locals()
