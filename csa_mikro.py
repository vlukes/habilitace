import sys
sys.path.append('.')
import numpy as nm
import fe2_mikro as nonlin_homog
from sfepy.discrete.evaluate import eval_equations
import sfepy.homogenization.coefs_base as cb
from sfepy.base.base import get_default


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
            names.append(f'_{ir}{ic}')
    return names, pis


class CorrDVel_d_pis_u(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)
        key = list(data.keys())[0]
        corr_sol = []
        dvel = data[key].state['v']
        for dv in dvel:
            clist, dout = create_pis(dv.reshape((-1, problem.get_dim())),
                                     vname='u')
            corr_sol.append(cb.CorrSolution(name=self.name, states=dout,
                                            components=clist))

        return cb.CorrSolution(name=f'list_{self.name}', states=corr_sol)


class CorrDVelocity(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        dim = self.dim
        requires = [req.split('|')[0] for req in self.requires]

        lab = '_' + self.corr_flag
        state_u = nm.array([data[requires[-1]].states[k]['u']
                            for k in cb.iter_sym(dim)])

        nsym = state_u.shape[0]
        out = state_u.reshape((nsym, -1, dim)).copy()

        if self.corr_flag == 'e':
            pis = nm.array([data[requires[0]].states[k]['u']
                            for k in cb.iter_sym(dim)])
            out += pis.reshape((nsym, -1, dim))

        corr_sol = cb.CorrSolution(name=self.name + lab,
                                   state={'v': out.reshape((nsym, -1))})

        if '|' in self.requires[-1]:
            multi_lab =  self.requires[-1].split('|')[1]
            lab += multi_lab

        nonlin_homog.multiproc_dependecies['dvelocity' + lab] = out

        return corr_sol


class CoefNonSymNonSymSA(cb.CoefNonSymNonSym):
    def __call__(self, volume, problem=None, data=None):
        pb = get_default(problem, self.problem)
        dim = pb.domain.mesh.dim
        for k in data.keys():
            if k.startswith('dvelocity'):
                nsym = data[k].state['v'].shape[0]
                pb.dvelocity = data[k].state['v'].reshape((nsym, -1, dim))
                break

        return cb.CoefNonSymNonSym.__call__(self, volume, problem, data)


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


def def_mat_diff(ts, coors, mode=None, term=None, problem=None, **kwargs):
    if mode != 'qp':
        return None

    sdelta = 1e-9
    dv = problem.dvelocity * 0.5 * sdelta
    tanmod1, stress1, _ = nonlin_homog.get_hyperelastic_mat(ts, coors, term,
                                                            problem, dv=dv)
    tanmod2, stress2, _ = nonlin_homog.get_hyperelastic_mat(ts, coors, term,
                                                            problem, dv=-dv)
    tanmod = (tanmod1 - tanmod2) / sdelta
    stress = (stress1 - stress2) / sdelta
    return {'sA': tanmod, 'sS': stress}


class CoefOneDVel(cb.CoefOne):
    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)

        term_mode = self.term_mode
        dvelocity0 = data['dvelocity'].state['v'].copy()

        coef = []
        for dvel in dvelocity0:
            problem.dvelocity = dvel.reshape((-1, problem.get_dim()))
            data['dvelocity'].state['v'] = dvel

            equations, variables = problem.create_evaluable(self.expression,
                                                            term_mode=term_mode)
            if hasattr(self, 'set_variables'):
                if isinstance(self.set_variables, list):
                    self.set_variables_default(variables, self.set_variables,
                                            data, self.dtype)
                else:
                    self.set_variables(variables, **data)

            val = eval_equations(equations, variables,
                                 term_mode=term_mode)

            coef.append(val / self._get_volume(volume))

        data['dvelocity'].state['v'] = dvelocity0

        return nm.array(coef)


class CoefSA1(cb.MiniAppBase):
    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)
        dvelocity = data['dvelocity'].state['v']
        isym = [ii for ii in cb.iter_nonsym(problem.get_dim())]

        nr = len(isym)
        nc = nr + 1 if hasattr(self, 'expression2') else nr

        coef = nm.zeros((len(dvelocity), nr, nc), dtype=self.dtype)
        # coef2 = nm.zeros((len(row), len(col)), dtype=self.dtype)

        for k, dvel in enumerate(dvelocity):
            problem.dvelocity = dvel.reshape((-1, problem.get_dim()))

            if hasattr(self, 'expression2'):
                stress = problem.evaluate(self.expression2)
                coef[k, :len(stress), -1] = stress

            pvars = problem.create_variables(['V', 'u', 'v'])
            pvars['V'].set_data(dvel)
            mtx = problem.evaluate(self.expression,
                                   mode='weak', dw_mode='matrix',
                                   var_dict={'V': pvars['V'], 'u': pvars['u'],
                                             'v': pvars['v']})

            for ir, (irr, icr) in enumerate(isym):
                ur = (data['pis_u'].states[irr, icr]['u']
                      + data['corrs_rs'].states[irr, icr]['u'])
                lmul = mtx.T.dot(ur)
                for ic, (irc, icc) in enumerate(isym):
                    uc = (data['pis_u'].states[irc, icc]['u']
                          + data['corrs_rs'].states[irc, icc]['u'])
                    coef[k, ir, ic] = (lmul * uc).sum()

        coef /= self._get_volume(volume)

        return coef


class CoefSA2(cb.MiniAppBase):
    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)
        isym = [ii for ii in cb.iter_nonsym(problem.get_dim())]

        nc = nr = len(isym)
        coef = nm.zeros((len(data['d_pis_u'].states), nr, nc), dtype=self.dtype)

        mtx = problem.evaluate(self.expression, mode='weak', dw_mode='matrix')

        for k, dvel in enumerate(data['d_pis_u'].states):
            for ir, (irr, icr) in enumerate(isym):
                ur = (data['pis_u'].states[irr, icr]['u']
                      + data['corrs_rs'].states[irr, icr]['u'])
                lmul = mtx.T.dot(ur)
                for ic, (irc, icc) in enumerate(isym):
                    uc = dvel.states[irc, icc]['u']
                    coef[k, ir, ic] = (lmul * uc).sum()

        coef /= self._get_volume(volume)

        return coef


def define(**kwargs):
    d = nonlin_homog.define(**kwargs)

    dvlist = []
    coefs = {}
    requirements = {}

    functions = {
        'mat_fce_diff': (def_mat_diff,),
    }

    # options = {'multiprocessing': False}

    materials = {
        'mat_he_diff': 'mat_fce_diff',
    }

    fields = {
        'dvelocity': ('real', 'vector', 'Y', 1),
    }

    variables = {
        'V': ('parameter field', 'dvelocity', '(set-to-None)'),
    }

    requirements.update({
        'dvelocity': {
            'requires': ['pis_u', 'corrs_rs'],
            'dim': d['dim'],
            'class': CorrDVelocity,
            'corr_flag': 'e',
        },
        'd_pis_u': {
            'requires': ['dvelocity'],
            'class': CorrDVel_d_pis_u,
        },
    })
    coefs.update({
        'sA1': {
            'requires': ['pis_u', 'corrs_rs', 'dvelocity'],
            'expression': 'de_sd_lin_elastic.i.Y(mat_he.A, v, u, V)',
            'class': CoefSA1,
        },
        'sA2': {
            'requires': ['pis_u', 'corrs_rs', 'd_pis_u'],
            'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, v, u)',
            'class': CoefSA2,
        },
        'divV': {
            'requires': ['dvelocity'],
            'expression': 'ev_div.i.Y(V)',
            'set_variables': [('V', 'dvelocity', 'v')],
            'class': CoefOneDVel,
        },
        'sAS': {
            'requires': ['pis_u', 'corrs_rs', 'dvelocity'],
            'expression': 'dw_nonsym_elastic.i.Y(mat_he_diff.sA, v, u)',
            'expression2': 'ev_integrate_mat.i.Y(mat_he_diff.sS, u)',
            'class': CoefSA1,
        },
    })

    return merge_locals(locals(), d)
