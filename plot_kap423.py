import os.path as osp
import numpy as nm
from scipy.io import loadmat
from plot_utils import plot

fig_format = 'png'

fname_hom = 'output_poropiezo/poropiezo_data_1D_nl_20.mat'
fname_dns = 'output_poropiezo_dns/poropiezo_data_dir_1D_nl_20.mat'


def plot_probe(title, ylabel, labs, x, y, vname, fname, lp='l_f', ylim=None):
    plot('', r'$%s \mathrm{[-]}$' % lp, ylabel, labs, x, y,
         fname, figsize=[5, 3], ylim=ylim)


odir = osp.split(fname_hom)[0]

label = 'non-linear' if '_nl_' in fname_hom else 'linear'

data_dns = loadmat(fname_dns)
print(f'data file: {fname_dns}')
time_dns = data_dns['time'].squeeze()

data_hom = loadmat(fname_hom)
print(f'data file: {fname_hom}')
time_hom = data_hom['time'].squeeze()

reg_dict = {'Left': r'$\Gamma_L$', 'Right': r'$\Gamma_R$',
            'Middle': r'$\Gamma_M$'}

lims = []
for k in reg_dict.keys():
    kr = f'fluxw_{k}'
    kh = f'fluxw_{k}'
    valr = nm.cumsum(data_dns[kr].squeeze())
    valh = nm.cumsum(data_hom[kh].squeeze())
    lims.append(nm.min([valr.min(), valh.min()]))
    lims.append(nm.max([valr.max(), valh.max()]))

lims = [nm.min(lims) * 1.1, nm.max(lims)*1.1]

for k in reg_dict.keys():
    kr = f'fluxw_{k}'
    kh = f'flux_{k}'
    valr = nm.cumsum(data_dns[kr].squeeze())
    valh = nm.cumsum(data_hom[kh].squeeze())
    fname = osp.join(odir, f'fig_dns_vs_hom_Q_{k}.{fig_format}')
    plot(f'Cumulative flux through {reg_dict[k]} - {label}',
         '$t$ [s]', '$Q_{%s}$ [m$^3$ / m$^2$]' % k[0],
         ['$Q^{DNS}$', '$Q^{HOM}$'],
         [time_dns, time_hom], [valr, valh],
         fname, figsize=[5, 3], ylim=lims)


xd = data_dns['lc'].squeeze()[0]
xh = data_hom['lc'].squeeze()[0]

timed = data_dns['time'].squeeze()
timeh = data_hom['time'].squeeze()

times = nm.array([0.25, 0.5, 0.75])
idxs = nm.asarray(nm.round(times / timed[1]), dtype=nm.int32)

labs = [r'$%s^{DNS}$', r'$%s^{HOM}$']

lims = {'pressure': [], 'velocity': [], 'displ': []}
for idx in idxs:
    for k, v in lims.items():
        if k == 'pressure':
            data = data_hom[k + '_rec'][idx, :]
        else:
            data = data_hom[k + '_rec'][idx, :, 0]

        v.append(data.min())
        v.append(data.max())

lims = {k: [nm.min(v), nm.max(v)] for k, v in lims.items()}

for idx in idxs:
    oflag = f'_ti{idx}'
    fname = osp.join(odir, f'fig_dns_vs_hom_p{oflag}.{fig_format}')
    plot_probe(r'Fluid pressure', r'$p\,\mathrm{[Pa]}$', [k % 'p' for k in labs],
               [xd, xh], [data_dns['pressure'][idx, :], data_hom['pressure_rec'][idx, :]],
               'pressure', fname, ylim=lims['pressure'])

    fname = osp.join(odir, f'fig_dns_vs_hom_w{oflag}.{fig_format}')
    plot_probe(r'Fluid velocity', r'$w_1\,\mathrm{[m/s]}$', [k % 'w_1' for k in labs],
               [xd, xh], [data_dns['velocity'][idx, :, 0], data_hom['velocity_rec'][idx, :, 0]],
               'velocity', fname, ylim=lims['velocity'])

    fname = osp.join(odir, f'fig_dns_vs_hom_u{oflag}.{fig_format}')
    plot_probe(r'Solid displacement', r'$u_1\,\mathrm{[m]}$', [k % 'u_1' for k in labs],
               [xd[:-1], xh], [data_dns['displ'][idx, :-1, 0], data_hom['displ_rec'][idx, :, 0]],
               'displacement', fname, lp='l_e', ylim=lims['displ'])

fname = osp.join(odir, f'fig_bolus.{fig_format}')
plot(r'Electric potential', r'$x_1\,\mathrm{[m]}$', r'$\varphi^2(t, x_1)\,\mathrm{[V]}$',
     [r'$t = %.2f$\,s' % v for v in times],
     [xd * 0.1, xd * 0.1, xd * 0.1],
     [data_dns['bolus'][idx, 0, :] for idx in idxs],
     fname, figsize=[5, 3])
