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

COLORS = {'normal': '#65758B', 'non_normal': '#007E80'}
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

    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelsize':10,'axes.titlesize':11,'axes.titleweight':'bold',
                         'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
                         'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'}):
        fig, axs = plt.subplots(2,2,figsize=(10.6,7.4))
        fig.subplots_adjust(left=.09,right=.96,bottom=.12,top=.84,wspace=.33,hspace=.48)
        fig.suptitle('Same eigenvalues, different transient responses',x=.09,y=.965,ha='left',fontsize=17,fontweight='bold')
        fig.text(.09,.915,r'Both maps have eigenvalues 0.8 and 0.5; directed coupling $F_{12}=$'+f'{coupling:g} in the non-normal map.',fontsize=10,color='#475569')
        a,b,c,d = axs.flat
        theta=np.linspace(0,2*np.pi,361)
        a.plot(np.cos(theta),np.sin(theta),color='#A0A9B3',lw=1.2,ls='--',label='Unit circle')
        a.axhline(0,color='#CBD5E1',lw=.8);a.axvline(0,color='#CBD5E1',lw=.8)
        for v in (.5,.8):
            a.scatter(v,0,s=75,facecolors='white',edgecolors=COLORS['normal'],linewidths=2,zorder=3)
            a.scatter(v,0,s=24,color=COLORS['non_normal'],zorder=4)
        a.annotate('0.5',(.5,0),xytext=(0,12),textcoords='offset points',ha='center')
        a.annotate('0.8',(.8,0),xytext=(0,-20),textcoords='offset points',ha='center')
        a.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),xlabel='Real part',ylabel='Imaginary part')
        a.set_aspect('equal',adjustable='box');a.set_title('(a) Identical stable spectra',loc='left')
        a.text(.02,.02,r'$\rho(F)=0.8$ for both',transform=a.transAxes,color='#475569',fontsize=9)
        b.plot(np.cos(theta),np.sin(theta),color='#A0A9B3',lw=1,ls='--')
        for name,xy in trajectories.items():
            b.plot(xy[:,0],xy[:,1],'-o',ms=3,lw=1.9,color=COLORS[name],label=LABELS[name])
            for j in (0,2):
                b.annotate('',xy=xy[j+1],xytext=xy[j],arrowprops=dict(arrowstyle='->',color=COLORS[name],lw=1.3))
        b.scatter(*x0,s=50,facecolors='white',edgecolors='#172B3A',zorder=5)
        b.annotate('Same initial state',x0,xytext=(6,8),textcoords='offset points',fontsize=8)
        b.scatter(0,0,s=20,c='#172B3A',zorder=4)
        b.set(xlabel=r'State $x_1$',ylabel=r'State $x_2$')
        b.set_aspect('equal',adjustable='box');b.set_title('(b) Different state trajectories',loc='left')
        # Zoom to the occupied quadrant; the dashed arc marks unit norm.
        xymax=max(1.1,max(float(np.abs(v).max()) for v in trajectories.values())+.15)
        b.set(xlim=(-.08,xymax+.12),ylim=(-.08,xymax+.12))
        b.text(.98,.06,'Dashed arc: unit norm',transform=b.transAxes,ha='right',fontsize=8,color='#475569')
        for name in maps:
            c.plot(steps,gains[name],'-o',color=COLORS[name],lw=2,ms=4,label=LABELS[name])
            c.plot(steps,np.linalg.norm(trajectories[name],axis=1),'--',color=COLORS[name],lw=1.1,alpha=.75)
            d.plot(times,variance[name]/2,color=COLORS[name],lw=2)
        c.axhline(1,color='#8794A1',ls=':',lw=1.2)
        c.set(xlabel='Discrete time step k',ylabel='Euclidean norm gain',xticks=[0,2,4,6,8,10,12],ylim=(0,max(1.15,float(gains['non_normal'].max())*1.15)))
        c.set_title('(c) Optimal gain vs. one fixed input',loc='left')
        c.text(.98,.92,'Solid: optimal at each step\nDashed: shared input',transform=c.transAxes,ha='right',va='top',fontsize=8,color='#475569')
        c.annotate(f"{summary['non_normal']['sigma']:.3f}", (1,gains['non_normal'][1]),xytext=(10,12),textcoords='offset points',color=COLORS['non_normal'],fontsize=9)
        d.axhline(1,color='#8794A1',ls=':',lw=1.2)
        d.set(xlabel='Continuous time (sample intervals)',ylabel=r'Normalized variance response $A(t)/A(0)$',xlim=(0,12),ylim=(0,max(1.08,float(variance['non_normal'].max()/2)*1.1)))
        d.set_title('(d) Isotropic impulse response',loc='left')
        d.text(.98,.82,rf"Non-normal: $R_A={summary['non_normal']['R_A']:.3f}$"+'\n'+rf"$R_\sigma={summary['non_normal']['R_sigma']:.3f}$",transform=d.transAxes,ha='right',va='top',color=COLORS['non_normal'],fontsize=10)
        for ax in (c,d): ax.grid(axis='y',alpha=.18)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([0],[0],color=COLORS[n],lw=2.5) for n in maps], [LABELS[n] for n in maps],loc='lower center',bbox_to_anchor=(.5,.016),ncol=2,frameon=False)
        for ext in ('png','svg'):
            fig.savefig(out/f'comparison.{ext}',dpi=200)
        plt.close(fig)
    print('Saved figure, diagnostic summary and response CSVs to',out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/examples/01_stability'))
    parser.add_argument('--coupling',type=float,default=0.6,help='Off-diagonal coupling in the non-normal map (default 0.6).')
    args=parser.parse_args()
    main(args.out,args.coupling)
