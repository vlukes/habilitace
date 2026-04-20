from functools import partial
import os.path as osp
import numpy as nm
import meshio
from scipy.spatial import cKDTree
from matplotlib import pyplot as mpl
import vtk
import pyvista as pv

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}",
        # Enforce default LaTeX font.
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)

ls_ = ['b-', 'y--', 'g:', 'm-.']


def plot(title, xlabel, ylabel, varlab, pars, vals, filename_results,
         ls=None, lw=2, marker=None, markersize=3, xticks=None, yticks=None,
         fontsize=14, figsize=None, ticklabel_style='sci',
         xinvert=False, yinvert=False,
         xlim=None, ylim=None, plot_style=None, color=None,
         lloc='best',
         xtick_labels=None, ytick_labels=None, close=True):
    if figsize is not None:
        fig = mpl.figure(figsize=figsize)
    else:
        fig = mpl.figure()
    mpl.clf()
    mpl.title(title, fontsize=fontsize)
    mpl.ylabel(ylabel, fontsize=fontsize)
    mpl.xlabel(xlabel, fontsize=fontsize)
    mpl.grid(True)
    mpl.ticklabel_format(style=ticklabel_style, axis='y', scilimits=(0, 0))
    fplot = {None: mpl.plot, 'logy': mpl.semilogy,
             'logx': mpl.semilogx, 'logxy': mpl.loglog}[plot_style]
    if isinstance(varlab, list) or isinstance(varlab, tuple):
        ls = ls_ if ls is None else ls
        ls0 = [ii[1:] for ii in ls]
        col = [ii[0] for ii in ls] if color is None else color
        for ii in range(len(varlab)):
            marker0 = '' if marker is None else marker[ii]
            x = pars[ii] if len(pars) == len(vals) else pars
            fplot(x, vals[ii], label=varlab[ii], lw=lw,
                  ls=ls0[ii], color=col[ii],
                  marker=marker0, markersize=markersize)
        if varlab.count(None) != len(varlab):
            mpl.legend(loc=lloc, fontsize=fontsize)

    else:
        ls = 'b-' if ls is None else ls
        ls0 = ls[1:]
        col = ls[0] if color is None else color
        fplot(pars, vals, label=varlab, lw=2,
              ls=ls0, color=col, marker=marker, markersize=markersize)

    ax = mpl.gca()
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        mpl.xlim(xlim)
    if ylim is not None:
        mpl.ylim(ylim)
    if xinvert:
        ax.invert_xaxis()
    if yinvert:
        ax.invert_yaxis()
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)

    mpl.tight_layout()

    fname, fext = osp.splitext(filename_results)
    exts = [fext, '.png'] if fext == '.pdf' else [fext]

    for ext in exts:
        fn = fname + ext
        print('saving figure to %s' % fn)
        fig.savefig(fn, dpi=200, bbox_inches='tight')

    if close:
        mpl.close(fig)
        return None
    else:
        return fig


def plot_mesh(mesh, label, svar, fname_out, du=None,
              zoom=1, scale=1, show_init=False, show_edges=True,
              cell_to_points=True, position=None, clim=None):
    if du is not None:
        grid0 = pv.from_meshio(mesh).outline()
        coors0 = mesh.points.copy()
        mesh.points = coors0 + du
    
    grid = pv.from_meshio(mesh)
    if cell_to_points:
        grid = grid.cell_data_to_point_data()

    if position is not None:
        pos_x, pos_y = position
    else:
        pos_x, pos_y = None, None

    plotter = pv.Plotter(off_screen=True)
    if show_init:
        plotter.add_mesh(grid0, color='#00000')
    plotter.add_mesh(grid, show_edges=show_edges, scalars=svar, show_scalar_bar=False)
    plotter.add_scalar_bar(label, width=0.4, n_labels=2,
                           title_font_size=30, label_font_size=30,
                           position_x=pos_x, position_y=pos_y)
    if clim is not None:
        plotter.update_scalar_bar_range(clim=clim)

    plotter.view_xy()
    plotter.camera.zoom(zoom)
    plotter.image_scale = scale
    plotter.show(screenshot=fname_out)
    print(f'  -> {fname_out}')
    del plotter

    if du is not None:
        mesh.points = coors0


def plot_macro_coefs(x, ys, pts_lab, data_dir, fig_format,
                     components, title, coef_names, coef_symbs, flag=''):
    for cset in components:
        for clab, cidxs in cset:
            if len(clab) > 4 and clab[-4] in '1234':
                cname, ci = clab[:-4], clab[-4:]
            elif len(clab) > 2 and clab[-2] in '1234':
                cname, ci = clab[:-2], clab[-2:]
            else:
                cname, ci = clab, None

            comp = coef_symbs[cname] % ci
            y = ys[cname][:, :, cidxs[0], cidxs[1]]
            varlab = ['$' + comp + r'(\hat{x}_{' + pts_lab[k] + '})$' for k in range(len(pts_lab))]

            fname_out = osp.join(data_dir, f'fig_{pts_lab}_{clab}{flag}.{fig_format}')
            plot(title=title % comp,
                figsize=(5, 3),
                xlabel='Loading step',
                ylabel=coef_names[cname] % comp,
                varlab=varlab, pars=x, vals=y,
                filename_results=fname_out)


def get_ss_mag(val):
    return nm.linalg.norm(nm.hstack([val, val[:, [-1]]]), axis=1)


def get_vtk_fnames(dname, fname, d=None):
    fname = osp.join(dname, fname)
    fmt = '{0:d}' if d is None else '{0:0=' + str(d) + 'd}'

    out = []
    flag = 0
    for k in range(10000):
        fn = f'{fname}.{fmt.format(k)}.vtk'
        if osp.isfile(fn):
            out.append(fn)
            flag = 0
        else:
            flag += 1

        if flag > 1000:
            break
    
    return out


def parse_log(f):
    inside = False
    rho, nnodes, time, neval, displ = [], [], [], [], []

    for line in f:
        if not inside and line.startswith("##############"):
            inside = True
            nflag = False
        elif inside and 'delta=' in line:
            # import pdb; pdb.set_trace()
            line = line.strip().strip('[]').strip("'").strip('"')
            idxs = line.find('delta=') + 6
            aux = line[idxs:].split(',')[0]
            if aux == 'None':
                # rho.append(nm.nan)
                rho.append(12e12)
            else:
                rho.append(float(aux))
        elif inside and line.startswith("sfepy: macro displacements:"):
            ldispl = []
        elif inside and line.startswith("sfepy: solved in "):
            time.append(float(line.split()[-2]))
        elif inside and line.startswith("micro:   number of vertices: ") and not nflag:
            nnodes.append(int(line.split()[-1]))
            nflag = True
        elif inside and line.startswith("sfepy:  ") and 'nd=' in line and 'step=9' in line:
            vals = line.split(':')[-1].strip()[1:-1]
            ldispl.append([float(k) for k in vals.split()])
        elif inside and line.startswith("sfepy: >>> num. mat. eval: "):
            neval.append(int(line.split()[-1]))
            inside = False
            print(rho[-1], nnodes[-1], time[-1], neval[-1])
            displ.append(ldispl)

    srho = sorted(list(set(rho)), reverse=True)
    snnodes = sorted(list(set(nnodes)))
    atime = nm.zeros((len(srho), len(snnodes)), dtype=nm.float64) + nm.nan
    aneval = nm.zeros((len(srho), len(snnodes)), dtype=nm.int32) + nm.nan
    displ = nm.array(displ)
    adispl = nm.zeros((len(srho), len(snnodes)) + displ.shape[1:], dtype=nm.int32) + nm.nan

    for irho, innodes, itime, ineval, idispl in zip(nm.array(rho), nnodes, time, neval, displ):
        i, j = srho.index(irho), snnodes.index(innodes)
        atime[i, j] = itime
        aneval[i, j] = ineval
        adispl[i, j] = idispl

    srho = [nm.nan if k == 12e12 else k for k in srho]

    return srho, snnodes, atime, aneval, adispl


def _merge_nodes_tab(mesh, ms_tab):
    if len(ms_tab) == 0:
        return

    remap = nm.ones((mesh.points.shape[0],), dtype=nm.int64)
    remap[ms_tab[:, 1]] = -1

    ndidxs = nm.where(remap > 0)[0]
    remap[ndidxs] = nm.arange(len(ndidxs))
    mesh.points = mesh.points[ndidxs, :]
    remap[ms_tab[:, 1]] = remap[ms_tab[:, 0]]

    pdata = mesh.point_data
    if pdata is not None:
        for k in pdata.keys():
            pdata[k] = pdata[k][ndidxs, ...]

    for cg in mesh.cells:
        cg.data = remap[cg.data]


def find_master_slave_nodes(coors, tol=1e-9):
    tr = cKDTree(coors)
    mtx = tr.sparse_distance_matrix(tr, tol).tocsr()
    nrow = nm.diff(mtx.indptr)
    idxs = nm.where(nrow > 1)[0]

    npairs_max = nm.sum(nrow[idxs] - 1)

    out = nm.empty((npairs_max, 2), dtype=nm.int64)
    idx0 = 0
    for ii in idxs:
        i1, i2 = mtx.indptr[ii], mtx.indptr[ii + 1]
        cols = mtx.indices[i1:i2]
        if cols[cols < ii].shape[0] == 0:
            nc = cols.shape[0]
            if nc == 2:
                out[idx0, :] = cols
                idx0 += 1
            else:
                idx1 = idx0 + nc - 1
                out[idx0:idx1, 0] = cols[0]
                out[idx0:idx1, 1] = cols[1:]
                idx0 = idx1

    return out[:idx0, :]


def merge_nodes(mesh, tol=1e-9):
    if '_not_merge' in mesh.point_data:
        idxs = nm.where(nm.logical_not(mesh.point_data['_not_merge']))[0]
        ms_tab = find_master_slave_nodes(mesh.points[idxs], tol=tol)
        ms_tab = idxs[ms_tab]
    else:
        ms_tab = find_master_slave_nodes(mesh.points, tol=tol)

    _merge_nodes_tab(mesh, ms_tab)


def mesh2qp(mesh, nqp):
    def displ_fun(du, cmap, conn):
        du1 = du.copy()
        for k, cep in enumerate(cmap):
            if isinstance(cep, tuple):
                lcep, mul = cep
            else:
                lcep, mul = cep, 0.5

            lconn = conn[:, lcep]
            du1k = du1[lconn[:, 0]] * (1 - mul) + du1[lconn[:, 1]] * mul
            du1 = nm.vstack([du1, du1k])
        
        return du1

    cremap = {
        '4_4': nm.array([
            [0, 5, 4, 8],
            [5, 1, 6, 4],
            [4, 6, 2, 7],
            [8, 4, 7, 3],
        ]),
        '4_3': nm.array([
            [4, 1, 2, 5],
            [6, 7, 5, 3],
            [0, 4, 7, 6],
        ]),
        '8_6': nm.array([
            [0, 1, 14, 13, 9, 10, 16, 15],
            [13, 14, 2, 3, 15, 16, 11, 12],
            [9, 19, 8, 15, 4, 21, 23, 17],
            [19, 10, 16, 8, 21, 5, 18, 23],
            [15, 8, 20, 12, 17, 23, 22, 7],
            [8, 16, 11, 20, 23, 18, 6, 22],
        ]),
    }

    cepoint = {
        '4_4': [
            [0, 2],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ],
        '4_3': [
            (nm.array([0, 1]), 0.66),
            (nm.array([2, 3]), 0.33),
            (nm.array([0, 3]), 0.5),
            (nm.array([4, 5]), 0.5),
        ],
        '8_6': [
            nm.arange(8),
            nm.array([0, 4]),
            nm.array([1, 5]), # 10
            nm.array([2, 6]),
            nm.array([3, 7]),
            nm.array([0, 3]),
            nm.array([1, 2]),
            nm.array([0, 3, 7, 4]), # 15
            nm.array([1, 2, 6, 5]),
            nm.array([4, 7]),
            nm.array([5, 6]),
            nm.array([0, 1, 5, 4]),
            nm.array([2, 3, 7, 6]), # 20
            nm.array([4, 5]),
            nm.array([6, 7]),
            nm.array([4, 5, 6, 7]),
        ],
    }

    el_type = {
        3: 'triangle',
        4: 'quad',
        8: 'hexahedron',
    }

    conn0 = mesh.cells[0].data 
    coors0 = mesh.points
    n_el, ne_nd = conn0.shape
    ekey = f'{ne_nd}_{nqp}'

    ne_ndd = len(cepoint[ekey])
    conn1 = nm.zeros((n_el, ne_nd + ne_ndd), dtype=nm.int64)
    conn1[:, :ne_nd] = conn0
    coors1 = coors0.copy()
    for k, cep in enumerate(cepoint[ekey]):
        # coors1k = nm.average(coors0[conn0[:, cep], :], axis=1)
        if isinstance(cep, tuple):
            lcep, mul = cep
        else:
            lcep, mul = cep, 0.5

        lconn = conn1[:, lcep]
        coors1k = coors1[lconn[:, 0]] * (1 - mul) + coors1[lconn[:, 1]] * mul
        conn1[:, ne_nd + k] = (nm.arange(len(coors1k)) + len(coors1))
        coors1 = nm.append(coors1, coors1k, axis=0)

    ne_nd1 = cremap[ekey].shape[1]
    conn = meshio.CellBlock(el_type[ne_nd1],
                            conn1[:, cremap[ekey]].reshape((-1, ne_nd1)))

    return (meshio.Mesh(points=coors1, cells=[conn]),
            partial(displ_fun, cmap=cepoint[ekey], conn=conn1))


# if __name__ == '__main__':
#     m = meshio.read('meshes/macro_L0.mesh')
#     m.write('mesh2qp_test0.vtk', binary=False)
#     m2, _ = mesh2qp(m, 4)
#     # import pdb; pdb.set_trace()
#     merge_nodes(m2)
#     m2.write('mesh2qp_test.vtk', binary=False)
#     m2.write('mesh2qp_test.mesh')