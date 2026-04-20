import os.path as osp
import numpy as nm
import meshio
from scipy.io import loadmat
import matplotlib.pyplot as plt
from plot_utils import get_vtk_fnames, mesh2qp, plot_mesh

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}",
        # Enforce default LaTeX font.
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

# generate data: sfepy-run csa_makro.py -d "delta=0.005,output_dir='output_fe2_b',save_qp=True"

# fig_format = 'pdf'
fig_format = 'png'
plot_steps = [1, 5, 9]
data_dir, rho, nbins = 'output_fe2_b', 0.005, 6
data_dir, rho, nbins = 'output_fe2_c', 0.001, 4
fname_fe2 = get_vtk_fnames(data_dir, 'macro_L')
fname_fe2_qp = osp.join(data_dir, 'cf_qp.mat')

data_qp = loadmat(fname_fe2_qp)

new_cent = nm.hstack([
    data_qp['step_iter'],
    data_qp['ncentroids'][:, 1:].T,
])

new_cent = nm.vstack([[0, 0, 1], new_cent])

steps = nm.hstack([[0], nm.where(nm.diff(new_cent[:, 0]))[0] + 1])

labels_ = nm.arange(new_cent.shape[0])
fig, ax = plt.subplots(figsize=[5,3])

label = 'loading step'
for x in steps:
    plt.axvline(x, color='silver', label=label, alpha=0.5, ls='-')
    label = None

ax.bar(labels_, new_cent[:, 2].cumsum())
ax.set_title(rf'Total number of centroids: $\rho = {rho}$')
ax.set_ylabel('Number of centroids', fontsize=12)
ax.set_xlabel('Macroscopic iteration', fontsize=12)
ax.legend()
plt.tight_layout()
fname_out = osp.join(data_dir, f'num_centroids_cum_{rho}.{fig_format}')
print(f'  -> {fname_out}')
fig.savefig(fname_out, dpi=300)

fig, ax = plt.subplots(figsize=[5,3])
label = 'loading step'
for x in steps:
    plt.axvline(x, color='silver', label=label, alpha=0.5, ls='-')
    label = None

ax.bar(labels_, new_cent[:, 2])
ax.set_title(rf'Number of new centroids: $\rho = {rho}$')
ax.set_ylabel('Number of new centroids', fontsize=12)
ax.set_xlabel('Macroscopic iteration', fontsize=12)
ax.legend()
plt.tight_layout()
fname_out = osp.join(data_dir, f'num_centroids_new_{rho}.{fig_format}')
print(f'  -> {fname_out}')
fig.savefig(fname_out, dpi=300)

for step in plot_steps:
    fn = fname_fe2[step]
    print(fn)
    mesh = meshio.read(fn)

    cid = []
    ncid = []
    for k in range(100):
        key = f'cid{k}'
        if key in mesh.cell_data:
            cid.append(mesh.cell_data[key][0])
            ncid.append(mesh.cell_data[f'ncid{k}'][0])
        else:
            break

    nqp = len(cid)

    cid = nm.stack(cid).T.reshape((-1))
    ncid = nm.stack(ncid).T.reshape((-1))

    fig, ax = plt.subplots(figsize=[5,3])
    label = 'loading step'
    ncid_max = ncid.max()
    # bins = nm.arange(ncid_max + 1) + 0.5
    bins = nm.arange(nbins + 1) + 0.5
    ax.hist(ncid, bins=bins, rwidth=0.9)
    ax.set_title(rf'Num. of centroids involved in approximation: $\rho = {rho}$, step ${step}$')
    ax.set_xlabel('Number of involved centroids', fontsize=12)
    ax.set_ylabel('Number of quadrature points', fontsize=12)
    ax.set_xticks(nm.arange(ncid_max) + 1)
    plt.tight_layout()
    fname_out = osp.join(data_dir, f'num_centroids_hist_{rho}_{step}.{fig_format}')
    # fname_out = osp.join(data_dir, f'fig_num_centroids_hist_{rho}_{k}.{fig_format}')
    print(f'  -> {fname_out}')
    fig.savefig(fname_out, dpi=300)

    meshX, displ_fun = mesh2qp(mesh, nqp)
    meshX.cell_data['nCid'] = [ncid]

    fname_out = osp.join(data_dir, f'macro_ncid_{rho}_{step}.png')
    plot_mesh(meshX, f'step {step}: n_centroids', 'nCid', fname_out,
              du=displ_fun(mesh.point_data['u']),
              zoom=1.4, scale=2, cell_to_points=False,
              position=(0.05,0.85),clim=[0, ncid.max()])
