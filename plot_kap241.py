import os.path as osp
import numpy as nm
from scipy.io import loadmat
import meshio
import matplotlib.pyplot as plt
import vtk
import pyvista as pv
from plot_utils import plot, mesh2qp, get_vtk_fnames

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
n_step = 80
nqp = 4
pt_ABC = nm.array([(18, 0), (72, 0), (98, 0)])

data_dir_fe2 = 'output_fe2'
data_dir_dns = 'output_dns'

fname_fe2_qp = osp.join(data_dir_fe2, 'cf_qp.mat')
fname_fe2 = get_vtk_fnames(data_dir_fe2, 'macro_16x8')[-1]
fname_dns = get_vtk_fnames(data_dir_dns, 'dns_32x16', d=2)

eid_ABC = [k // 16 * 2 * 32 + k % 16 * 2 for k, _ in pt_ABC]

data_fe2 = loadmat(fname_fe2_qp)
pt_ABC_qp = pt_ABC[:, 0] * nqp + pt_ABC[:, 1]

stress_dns = []
for fn in fname_dns[::(len(fname_dns) // (n_step - 1))]:
    print(fn)
    m = meshio.read(fn)
    eid = m.cell_data['eid'][0]
    astress = m.cell_data['acauchy_stress'][0]
    stress_dns.append([nm.mean(astress[eid == k], axis=0) for k in eid_ABC])

stress_dns = nm.array(stress_dns)
stress_dns_mag = nm.linalg.norm(stress_dns, axis=2)

components = ([[('S11', (0, 0)), ('S22', (1, 0)), ('S12', (2, 0))]])
title = r'Aver. Cauchy stress $%s$ at points A, B, C'
coef_names = {'S': 'Aver. Cauchy stress $%s$ [Pa]'}
title_diff = r'Relative error of $%s$ at points A, B, C'
coef_names_diff = {'S': r'Relative error $Err^{rel}_{\mathcal{S}_{%s}}$ [-]'}
coef_symb = {'S': r'{\mathcal{S}_{%s}}'}

pt_ABC_lab = 'ABC'

for cset in components:
    for clab, cidxs in cset:
        if len(clab) > 4 and clab[-4] in '1234':
            cname, ci = clab[:-4], clab[-4:]
        elif len(clab) > 2 and clab[-2] in '1234':
            cname, ci = clab[:-2], clab[-2:]
        else:
            cname, ci = clab, None

        comp = coef_symb[cname] % ci

        steps = data_fe2['step_iter'][:, 0]
        idxs = nm.array(list(nm.where(nm.diff(steps))[0]) + [len(steps) - 1])
        
        x = [steps[idxs]] * len(pt_ABC_qp)
        y = data_fe2[cname][idxs, :, cidxs[0], cidxs[1]][:, pt_ABC_qp].T
        varlab = ['$' + comp + r'(\hat{x}_{' + pt_ABC_lab[k] + '})$' for k in range(len(pt_ABC_qp))]

        fname_out = osp.join(data_dir_fe2, f'fig_dir_ABC_{clab}.{fig_format}')
        print(f'  -> {fname_out}')
        plot(title=title % comp,
             figsize=(5, 3),
             xlabel='Loading step',
             ylabel=coef_names[cname] % comp,
             varlab=varlab, pars=x, vals=y,
             filename_results=fname_out)

        y2 = stress_dns[1:, :, cidxs[0]].T
        err = nm.abs(y - y2) / stress_dns_mag[1:, :].T
        varlab = [r'$\hat{x}_{' + pt_ABC_lab[k] + '}$' for k in range(len(pt_ABC_qp))]
        fname_out = osp.join(data_dir_fe2, f'fig_dir_ABC_rerr_{clab}.{fig_format}')
        print(f'  -> {fname_out}')
        plot(title=title_diff % comp,
             figsize=(5, 3),
             xlabel='Loading step',
             ylabel=coef_names_diff[cname] % ci,
             varlab=varlab, pars=x, vals=err,
             filename_results=fname_out)

for pt, eid in zip(pt_ABC_qp, eid_ABC):
    fn = get_vtk_fnames(data_dir_fe2, f'micro_2.recovered_{pt:03d}', d=5)[-1]
    print(fn)
    m1 = meshio.read(fn)
    m1.cell_data['stress'] = [nm.linalg.norm(m1.cell_data['cauchy_stress'][0], axis=1)]
    m1.cell_data['strain'] = [nm.linalg.norm(m1.cell_data['green_strain'][0], axis=1)]
    m1.points += m1.point_data['displacement']

    fn = fname_dns[-1]
    print(fn, eid)
    m2 = meshio.read(fn)
    m2.cell_data['stress'] = [nm.linalg.norm(m2.cell_data['cauchy_stress'][0], axis=1)]
    m2.cell_data['strain'] = [nm.linalg.norm(m2.cell_data['green_strain'][0], axis=1)]
    m2.points += m2.point_data['u']

    idxs = m2.cell_data['eid'][0] == eid
    max_stress = nm.max([m1.cell_data['stress'][0].max(),
                         m2.cell_data['stress'][0][idxs].max()])
    max_strain = nm.max([m1.cell_data['strain'][0].max(),
                         m2.cell_data['strain'][0][idxs].max()])

    grid = pv.from_meshio(m1)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True, scalars='stress', show_scalar_bar=False)
    plotter.add_scalar_bar(r'$\vert\tilde\sigma\vert$',
                           width=0.4, n_labels=2, title_font_size=30, label_font_size=30,
                           position_x=0.05, position_y=0)
    plotter.update_scalar_bar_range(clim=[0, max_stress])
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    fname_out = osp.join(data_dir_fe2, f'fig_dir_vs_mac_stress_mac_{pt}.png')
    print(f'  -> {fname_out}')
    plotter.show(screenshot=fname_out)
    del plotter

    grid = pv.from_meshio(m1)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True, scalars='strain', show_scalar_bar=False)
    plotter.add_scalar_bar(r'$\vert E\vert$',
                        width=0.4, n_labels=2, title_font_size=30, label_font_size=30,
                        position_x=0.05, position_y=0.0)
    plotter.update_scalar_bar_range(clim=[0, max_strain])
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    fname_out = osp.join(data_dir_fe2, f'fig_dir_vs_mac_strain_mac_{pt}.png')
    print(f'  -> {fname_out}')
    plotter.show(screenshot=fname_out)
    del plotter

    grid0 = pv.from_meshio(m2)
    grid = grid0.threshold((eid, eid), scalars='eid')

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True, scalars='stress', show_scalar_bar=False)
    plotter.add_scalar_bar(r'$\vert\sigma^{DNS}\vert$',
                        width=0.4, n_labels=2, title_font_size=30, label_font_size=30,
                        position_x=0.05, position_y=0)
    plotter.update_scalar_bar_range(clim=[0, max_stress])
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    fname_out = osp.join(data_dir_dns, f'fig_dir_vs_mac_stress_dir_{eid}.png')
    print(f'  -> {fname_out}')
    plotter.show(screenshot=fname_out)
    del plotter

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True, scalars='strain', show_scalar_bar=False)
    plotter.add_scalar_bar(r'$\vert E^{DNS}\vert$',
                        width=0.4, n_labels=2, title_font_size=30, label_font_size=30,
                        position_x=0.05, position_y=0)
    plotter.update_scalar_bar_range(clim=[0, max_strain])
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    fname_out = osp.join(data_dir_dns, f'fig_dir_vs_mac_strain_dir_{eid}.png')
    print(f'  -> {fname_out}')
    plotter.show(screenshot=fname_out)
    del plotter

scf = 1.

fn = fname_fe2
print(fn)
m1 = meshio.read(fn)
m1.cell_data['stress'] = [nm.linalg.norm(m1.cell_data['cauchy_stress'][0], axis=1)]
m1.points += m1.point_data['u'] * scf
# grid = pv.from_meshio(m1)

m1x, displ_fun = mesh2qp(m1, nqp)
# m1x.points += displ_fun(m1.point_data['u'])
m1x.cell_data['stress'] = [nm.linalg.norm(data_fe2['S'][-1], axis=1)]
# m1x.write('aux.vtk')
# import pdb; pdb.set_trace()
grid = pv.from_meshio(m1x)

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(grid, show_edges=False, scalars='stress', show_scalar_bar=False)
plotter.add_scalar_bar(r'$\vert\mathcal{S}\vert$',
                        width=0.4, n_labels=2, title_font_size=30, label_font_size=30,
                        position_x=0.1, position_y=0.05)
plotter.view_xy()
plotter.camera.zoom(1.45)

fname_out = osp.join(data_dir_fe2, 'fig_dir_vs_mac_homS.png')
plotter.show(screenshot=fname_out)
del plotter

fn = fname_dns[-1]
print(fn)
m2 = meshio.read(fn)
m2.cell_data['astress'] = [nm.linalg.norm(m2.cell_data['acauchy_stress'][0], axis=1)]
m2.points += m2.point_data['u'] * scf

grid = pv.from_meshio(m2)

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(grid, show_edges=False, scalars='astress', show_scalar_bar=False)
plotter.add_scalar_bar(r'$\vert S^{DNS}\vert$',
                        width=0.4, n_labels=2, title_font_size=30, label_font_size=30,
                        position_x=0.1, position_y=0.05)
plotter.view_xy()
plotter.camera.zoom(1.45)

fname_out = osp.join(data_dir_dns, 'fig_dir_vs_mac_dirS.png')
plotter.show(screenshot=fname_out)
del plotter
