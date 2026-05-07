import os.path as osp
import numpy as nm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import meshio
from plot_utils import (get_vtk_fnames, get_ss_mag,
                        plot_mesh, plot_macro_coefs)

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
nqp = 4
pt_ABC = nm.array([(0, 0), (22, 0), (27, 0)])
pt_ABC_lab = 'ABC'

data_dir = 'output_perf'

fname_fe2_qp = osp.join(data_dir, 'cf_qp_nonlinear_perf.mat')
fname_fe2 = get_vtk_fnames(data_dir, 'macro_L', d=3, format=1)

components = (
    [('S|11', (0, 0)), ('S|22', (1, 0)), ('S|12', (2, 0))],
    # [('p|1', (0, 0)), ('p|2', (0, 0)),],
    # [('Q|11', (0, 0)), ('Q|22', (1, 1)), ('Q|12', (0, 1))],
    # [('A|1111', (0, 0)), ('A|2222', (3, 3)), ('A|1122', (0, 3))],
    # [('B1|11', (0, 0)), ('B1|22', (1, 1)), ('B1|12', (0, 1))],
    # [('B2|11', (0, 0)), ('B2|22', (1, 1)), ('B2|12', (0, 1))],
    # [('C1|11', (0, 0)), ('C1|22', (1, 1)), ('C1|12', (0, 1))],
    # [('C2|11', (0, 0)), ('C2|22', (1, 1)), ('C2|12', (0, 1))],
)

title = r'$%s$ at quadrature points A, B, C'
coef_names = {
    'S': 'Aver. Cauchy stress $%s$ [Pa]',
    'p': 'Channel pressure $%s$ [Pa]',
    'Q': 'Retardation stress $%s$ [Pa]',
    'A': 'Homog. stiffness $%s$ [Pa]',
    'B1': 'Biot poroela. coef. $%s$ [-]',
    'B2': 'Biot poroela. coef. $%s$ [-]',
    'C1': 'Ch. permeab. $%s$ [$m^2 / (Pa' + r'\cdot' + ' s)$]',
    'C2': 'Ch. permeab. $%s$ [$m^2 / (Pa' + r'\cdot' + ' s)$]',
}
coef_symbs = {
    'S': r'{\mathcal{S}_{%s}}',
    'p': r'{p^0_{%s}}',
    'Q': r'{\mathcal{Q}_{%s}}',
    'A': r'{\mathcal{A}_{%s}}',
    'B1': r'{\mathcal{B}^1_{%s}}',
    'B2': r'{\mathcal{B}^2_{%s}}',
    'C1': r'{\mathcal{C}^1_{%s}}',
    'C2': r'{\mathcal{C}^2_{%s}}',
}

# coefficients
data_qp = loadmat(fname_fe2_qp)
times = data_qp['times'].ravel()
x = [times] * len(pt_ABC)
nt, nch, nel, nqp, _, _ = data_qp['pressure'].shape
pressure = data_qp['pressure'].reshape((-1, nch, nel * nqp, 1))

pt_ABC_qp = pt_ABC[:, 0] * nqp + pt_ABC[:, 1]

y = {k: data_qp[k][:, pt_ABC_qp].transpose((1, 0, 2, 3))
     for k in ['S', 'Q', 'A', 'B1', 'B2', 'C1', 'C2']}
y.update({'p': pressure[:, :, pt_ABC_qp].transpose((2, 0, 1, 3))})

plot_macro_coefs(x, y, pt_ABC_lab, data_dir, fig_format,
                 components, title, coef_names, coef_symbs,
                 xlabel='Time [s]')

# deforom. macro domain
mesh = meshio.read(fname_fe2[-1])
du = mesh.point_data['u']
fname_out = osp.join(data_dir, f'macro_u_{nt - 1}.png')
plot_mesh(mesh, r'$\vert {u}^0 \vert$', 'u', fname_out, du=du, zoom=1.4)

mesh.cell_data['S']= [get_ss_mag(mesh.cell_data['cauchy_stress'][0])]
fname_out = osp.join(data_dir, f'macro_S_{nt - 1}.png')
plot_mesh(mesh, r'$\vert\mathcal{S}\vert$', 'S', fname_out, du=du, zoom=1.4)

mesh.cell_data['S']= [get_ss_mag(mesh.cell_data['cauchy_stress'][0])]
fname_out = osp.join(data_dir, f'macro_S_{nt - 1}.png')
plot_mesh(mesh, r'$\vert\mathcal{S}\vert$', 'S', fname_out, du=du, zoom=1.4)

for ich in range(nch):
    sch = str(ich + 1)
    pvel = f'w{sch}'

    fname_out = osp.join(data_dir, f'macro_p{sch}.png')
    plot_mesh(mesh, r'$p^0_{' + f'{sch}'+  r'}$', f'p{sch}', fname_out, du=du,
              zoom=1.4)

    mesh.cell_data[f'{pvel}_mag']= [nm.linalg.norm(mesh.cell_data[pvel][0], axis=1)]
    fname_out = osp.join(data_dir, f'macro_w{sch}.png')
    plot_mesh(mesh, r'$\vert {w}^0_{' + f'{sch}'+  r'}\vert$', f'{pvel}_mag',
              fname_out, du=du, zoom=1.4, arrows=(pvel, 20))