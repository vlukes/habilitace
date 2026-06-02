import os.path as osp
from scipy.io import loadmat
from plot_utils import plot

fig_format = 'png'

fname_nlin = 'output_poroela/poroela_data_nl.mat'
fname_lin = 'output_poroela/poroela_data_l.mat'

odir = osp.split(fname_lin)[0]

data_lin = loadmat(fname_lin)
data_nlin = loadmat(fname_nlin)

x = [data_lin['time'].ravel(), data_nlin['time'].ravel()]

y = [data_lin['pressure'][:, 50], data_nlin['pressure'][:, 50]]
fname = osp.join(odir, f'fig_lin_vs_nlin_B_p.{fig_format}')
plot(r'Pressure at point B: linear $\times$ nonlinear model',
     r'time [s]', r'$p$ [Pa]',
     ['linear', 'nonlinear'],
     x, y,
     fname, figsize=[5, 3])

y = [data_lin['strain'][:, 50, 2], data_nlin['strain'][:, 50, 2]]
fname = osp.join(odir, f'fig_lin_vs_nlin_B_u.{fig_format}')
plot(r'Deformation at point B: linear $\times$ nonlinear model',
     r'time [s]', r'$e_{33}$ [-]',
     ['linear', 'nonlinear'],
     x, y,
     fname, figsize=[5, 3])
