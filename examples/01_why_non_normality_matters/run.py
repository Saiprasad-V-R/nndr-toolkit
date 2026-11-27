#!/usr/bin/env python3
"""Example 1: identical stable spectra, different finite-time responses."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import logm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nndr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _figure_style import configure_style, format_axis, panel_label, save_figure, BLUE, RED, GREEN

COLORS = {'normal': '#666666', 'non_normal': BLUE}
LABELS = {'normal': 'Normal', 'non_normal': 'Non-normal'}


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main(out, coupling=0.6):
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError('coupling must be finite and nonnegative')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dt = 1.0
    maps = {'normal': np.diag([0.8, 0.5]),
            'non_normal': np.array([[0.8, coupling], [0., 0.5]])}
    # Same initial perturbation for BOTH maps, selected by the non-normal map.
    x0 = nndr.snapshot_gain(maps['non_normal'], 1)['v'].copy()
    if x0[1] < 0:  # reproducible sign convention for the illustration
        x0 *= -1
    steps = np.arange(13)
    times = np.linspace(0., 12., 481)
    summary, gains, trajectories, variance = {}, {}, {}, {}
    for name, F in maps.items():
        d = nndr.threshold_diagnostics(F, dt=dt)
        L = logm(F) / dt
        if np.max(np.abs(L.imag)) > 1e-10:
            raise ValueError('Expected a real principal logarithm')
        L = L.real
        gains[name] = np.array([nndr.optimal_gain(F, int(k)) for k in steps])
        trajectories[name] = np.array([np.linalg.matrix_power(F, int(k)) @ x0 for k in steps])
        variance[name] = np.array([nndr.variance_response(L, t, continuous=True) for t in times])
        summary[name] = dict(rho=nndr.spectral_radius(F), sigma=nndr.sigma_max(F),
                             departure=nndr.departure_from_normality(F),
                             R_A=float(d['R_A']), R_sigma=float(d['R_sigma']),
                             stable_reduced=bool(d['stable_reduced']),
                             rate_status=d['rate_status'],
                             max_sampled_gain=float(gains[name].max()),
                             gain_peak_step=int(steps[gains[name].argmax()]))
        assert d['stable_reduced'] and d['rate_admissible']
        assert np.isclose(gains[name][1], summary[name]['sigma'])
        assert np.isclose(variance[name][0], 2.)
        assert np.all(np.linalg.norm(trajectories[name], axis=1) <= gains[name]+1e-12)
        assert (d['R_sigma'] > 1) == (summary[name]['sigma'] > 1) or np.isclose(d['R_sigma'],1)
    assert np.allclose(np.sort(np.linalg.eigvals(maps['normal'])),
                       np.sort(np.linalg.eigvals(maps['non_normal'])))
    if coupling == 0.6:
        assert summary['non_normal']['R_A'] < 1 < summary['non_normal']['R_sigma']
        assert np.all(np.diff(variance['non_normal']) <= 1e-10)
    write_csv(out/'discrete_response.csv',
              ['step','normal_optimal_gain','non_normal_optimal_gain','normal_x1','normal_x2',
               'non_normal_x1','non_normal_x2','normal_trajectory_norm','non_normal_trajectory_norm'],
              ((int(k), gains['normal'][i], gains['non_normal'][i], *trajectories['normal'][i],
                *trajectories['non_normal'][i], np.linalg.norm(trajectories['normal'][i]),
                np.linalg.norm(trajectories['non_normal'][i])) for i,k in enumerate(steps)))
    write_csv(out/'variance_response.csv', ['time','normal_A','non_normal_A'],
              zip(times, variance['normal'], variance['non_normal']))
    metadata = {'sample_interval':dt, 'coupling':coupling, 'initial_perturbation':x0.tolist(),
                'maps':{name:F.tolist() for name,F in maps.items()},
                'diagnostics':summary, 'nndr_version':getattr(nndr,'__version__','unknown')}
    (out/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Shared unit initial perturbation:', x0)

    with plt.rc_context():
        configure_style()
        fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
        fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.94, wspace=.40, hspace=.46)
        a,b,c,d = axs.flat
        theta = np.linspace(0, 2*np.pi, 361)
        a.plot(np.cos(theta), np.sin(theta), color='.65', ls='--', lw=.9)
        a.axhline(0, color='.8', lw=.7); a.axvline(0, color='.8', lw=.7)
        for v in (.5,.8):
            a.scatter(v,0,s=45,facecolors='white',edgecolors=COLORS['normal'],linewidths=1.1,zorder=3)
            a.scatter(v,0,s=13,color=COLORS['non_normal'],zorder=4)
        a.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),xlabel=r'$\mathrm{Re}(\lambda)$',ylabel=r'$\mathrm{Im}(\lambda)$')
        a.set_aspect('equal',adjustable='box')
        b.plot(np.cos(theta), np.sin(theta), color='.65', ls='--', lw=.9)
        for name, xy in trajectories.items():
            b.plot(xy[:,0],xy[:,1],'-o',ms=2.5,color=COLORS[name])
            for j in (0,2):
                b.annotate('',xy=xy[j+1],xytext=xy[j],arrowprops=dict(arrowstyle='->',color=COLORS[name],lw=.9))
        b.scatter(*x0,s=32,facecolors='white',edgecolors='black',zorder=5)
        b.scatter(0,0,s=12,c='black',zorder=4)
        xymax=max(1.1,max(float(np.abs(v).max()) for v in trajectories.values())+.15)
        b.set(xlabel=r'$x_1$',ylabel=r'$x_2$',xlim=(-.08,xymax+.12),ylim=(-.08,xymax+.12))
        b.set_aspect('equal',adjustable='box')
        for name in maps:
            c.plot(steps,gains[name],'-o',color=COLORS[name],ms=3)
            c.plot(steps,np.linalg.norm(trajectories[name],axis=1),'--',color=COLORS[name],lw=1.0)
            d.plot(times,variance[name]/2,color=COLORS[name])
        c.axhline(1,color='.5',ls=':',lw=.9)
        c.set(xlabel=r'Step $k$',ylabel='Norm gain',ylim=(0,max(1.15,float(gains['non_normal'].max())*1.15)))
        d.axhline(1,color='.5',ls=':',lw=.9)
        d.set(xlabel=r'Time $t$',ylabel=r'$A(t)/A(0)$',xlim=(0,12),ylim=(0,max(1.08,float(variance['non_normal'].max()/2)*1.1)))
        for ax,label in zip(axs.flat,'abcd'):
            format_axis(ax); panel_label(ax,label,x=-.15,y=1.02)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([0],[0],color=COLORS[n]) for n in maps], [LABELS[n] for n in maps],loc='lower center',bbox_to_anchor=(.5,.025),ncol=2)
        save_figure(fig,out,'comparison')
    print('Saved figure, diagnostic summary and response CSVs to',out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/examples/01_stability'))
    parser.add_argument('--coupling',type=float,default=0.6,help='Off-diagonal coupling in the non-normal map (default 0.6).')
    args=parser.parse_args()
    main(args.out,args.coupling)
