import os.path as osp
import numpy as nm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from plot_utils import (get_vtk_fnames, plot_macro_coefs)

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}",
        # Enforce default LaTeX font.
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

# fig_format = 'pdf'
fig_format = 'png'
nqp = 3
pt_ABC = nm.array([(18, 0), (72, 0), (98, 0)])
pt_ABC_lab = 'ABC'

data_dir_fe2 = 'output_fe2_perf'
data_dir_dns = 'output_dns_perf'

fname_fe2_qp = osp.join(data_dir_fe2, 'cf_qp_nonlinear_perf.mat')

components = ([('S|11', (0, 0)), ('S|22', (1, 0)), ('S|12', (2, 0))], [('p|1', (0, 0))])
title = r'$%s$ at quadrature points A, B, C'
coef_names = {
    'S': 'Aver. Cauchy stress $%s$ [Pa]',
    'p': 'Channel pressure $%s$ [Pa]',
}
coef_symbs = {
    'S': r'{\mathcal{S}_{%s}}',
    'p': r'{p^0_{%s}}',
}
# title_diff = r'Relative error of $%s$'
# coef_names_diff = {
#     'S': r'Err^{rel}_{\mathcal{S}_{%s}}',
#     'p1': r'Err^{rel}_{p^0_{%s}}',
# }

pt_ABC_qp = pt_ABC[:, 0] * nqp + pt_ABC[:, 1]

data_qp = loadmat(fname_fe2_qp)
times = data_qp['times'].ravel()
# import pdb; pdb.set_trace()
x = [times] * len(pt_ABC)
p1 = data_qp['pressure'][:, 0].reshape((-1, data_qp['pressure'].shape[2] * nqp, 1, 1))
y = {
    'S': data_qp['S'][:, pt_ABC_qp].transpose((1, 0, 2, 3)),
    'p': p1[:, pt_ABC_qp].transpose((1, 0, 2, 3)),
}
plot_macro_coefs(x, y, pt_ABC_lab, data_dir_fe2, fig_format,
                 components, title, coef_names, coef_symbs,
                 xlabel='Time [s]')
