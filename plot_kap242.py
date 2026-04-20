import os.path as osp
import numpy as nm
import meshio
from scipy.io import loadmat
import matplotlib.pyplot as plt
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

# generate data: sfepy-run fe2_makro.py -d "output_dir='output_fe2_a',recovery_idxs=[(0,0),(22,0),(27,0)],save_qp=True"

# fig_format = 'pdf'
fig_format = 'png'
n_step = 10
data_dir = 'output_fe2_a'
fname_fe2 = get_vtk_fnames(data_dir, 'macro_L')
fname_fe2_qp = osp.join(data_dir, 'cf_qp.mat')
fname_fe2_mic = 'micro_1.recovered'

nqp = 4
pt_ABC = nm.array([(0, 0), (22, 0), (27, 0)])
pt_ABC_lab = 'ABC'

components = (
    [('S11', (0, 0)), ('S22', (1, 0)), ('S12', (2, 0))],
    [('A1111', (0, 0)), ('A2222', (3, 3)), ('A1122', (0, 3))],
)
title = r'$%s$ at quadrature points A, B, C'
coef_names = {
    'S': 'Aver. Cauchy stress $%s$ [Pa]',
    'A': 'Homog. stiffness $%s$ [Pa]',
}
coef_symbs = {
    'S': r'{\mathcal{S}_{%s}}',
    'A': r'{\mathcal{A}_{%s}}',
}

pt_ABC_qp = pt_ABC[:, 0] * nqp + pt_ABC[:, 1]

data_qp = loadmat(fname_fe2_qp)
steps = data_qp['step_iter'][:, 0]
idxs = nm.array(list(nm.where(nm.diff(steps))[0]) + [len(steps) - 1])
x = [steps[idxs]] * len(pt_ABC)
y = {cn: data_qp[cn][idxs][:, pt_ABC_qp].transpose((1, 0, 2, 3))
     for cn in coef_names.keys()}

plot_macro_coefs(x, y, pt_ABC_lab, data_dir, fig_format,
                 components, title, coef_names, coef_symbs)

displ = []
nd0 = None

# load all time steps
for step, fn in enumerate(fname_fe2):
    print(fn)
    mesh = meshio.read(fn)

    if step == 0:
        d = (mesh.points[:, 0]**2 + mesh.points[:, 1]**2)
        nd0 = nm.argmax(d)
        coors0 = mesh.points.copy()

    displ.append(mesh.point_data['u'][nd0])

displ = nm.array(displ)

# displacement
fig, ax = plt.subplots(figsize=[5,4])
ax.set_title('Macroscopic displacaments at point D', fontsize=16)
ax.plot(displ[:, 0], 'b-', label=r'$u^0_1$')
ax.plot(displ[:, 1], 'r--', label=r'$u^0_2$')
ax.set_xlabel('Loading step', fontsize=16)
ax.set_ylabel(r'Macroscopic displacements', fontsize=16)
ax.grid(True)
ax.legend(fontsize=16)
plt.tight_layout()
fname_out = osp.join(data_dir, f'fig_D_u.{fig_format}')
print(f'  -> {fname_out}')
fig.savefig(fname_out, dpi=300)

du = mesh.point_data['u']
fname_out = osp.join(data_dir, f'macro_u_{n_step - 1}.png')
plot_mesh(mesh, r'$\vert {u}^0 \vert$', 'u', fname_out, du=du, zoom=1.4)

# Cauchy stress
for icomp, comp in enumerate(['11', '22', '12']):
    du = mesh.point_data['u']
    mesh.cell_data['S']= [mesh.cell_data['cauchy_stress'][0][:, icomp]]
    fname_out = osp.join(data_dir, f'macro_S{comp}_{n_step - 1}.png')
    plot_mesh(mesh, r'$\mathcal{S}_{' + comp + r'}$', 'S', fname_out, du=du, zoom=1.4)

du = mesh.point_data['u']
mesh.cell_data['S']= [get_ss_mag(mesh.cell_data['cauchy_stress'][0])]
fname_out = osp.join(data_dir, f'macro_S_{n_step - 1}.png')
plot_mesh(mesh, r'$\vert\mathcal{S}\vert$', 'S', fname_out, du=du, zoom=1.4)

# Green strain
for icomp, comp in enumerate(['11', '22', '12']):
    du = mesh.point_data['u']
    mesh.cell_data['E']= [mesh.cell_data['green_strain'][0][:, icomp]]
    fname_out = osp.join(data_dir, f'macro_E{comp}_{n_step - 1}.png')
    plot_mesh(mesh, r'$E_{' + comp + r'}$', 'E', fname_out, du=du, zoom=1.4)

du = mesh.point_data['u']
mesh.cell_data['E']= [get_ss_mag(mesh.cell_data['green_strain'][0])]
fname_out = osp.join(data_dir, f'macro_E_{n_step - 1}.png')
plot_mesh(mesh, r'$\vert E\vert$', 'E', fname_out, du=du, zoom=1.4)

# micro
for pt, ptid in zip(pt_ABC_qp, pt_ABC_lab):
    fn = get_vtk_fnames(data_dir, f'{fname_fe2_mic}_{pt:03d}', d=5)[-1]
    print(fn)
    mesh = meshio.read(fn)
    du = mesh.point_data['displacement']
    mesh.cell_data['S']= [get_ss_mag(mesh.cell_data['cauchy_stress'][0])]
    fname_out = osp.join(data_dir, f'micro_{ptid}_S_{n_step - 1}.png')
    plot_mesh(mesh, r'$\vert \tilde\sigma\vert$', 'S', fname_out, du=du,
              zoom=0.95, show_init=True, scale=2)

    mesh.cell_data['E']= [get_ss_mag(mesh.cell_data['green_strain'][0])]
    fname_out = osp.join(data_dir, f'micro_{ptid}_E_{n_step - 1}.png')
    plot_mesh(mesh, r'$\vert E\vert$', 'E', fname_out, du=du,
              zoom=0.95, show_init=True, scale=2)