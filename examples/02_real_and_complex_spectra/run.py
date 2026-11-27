#!/usr/bin/env python3
"""Compare two spectral families and the effect of sampling interval."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nndr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _figure_style import configure_style, format_axis, panel_label, save_figure, BLUE, RED, GREEN

FAMILIES = {'real': lambda k: nndr.real_generator(1., 3., k),
            'complex': lambda k: nndr.complex_generator(1., 2., k)}
TITLES = {'real': 'Real spectrum', 'complex': 'Complex spectrum'}
LEVELS = (0., 1.5, 3.5)
COLORS = ('#666666', BLUE, RED)
RA_COLOR, RS_COLOR = BLUE, RED


def diagnostics(family, k, dt):
    L = FAMILIES[family](k)
    F = expm(L*dt)
    d = nndr.threshold_diagnostics(F, dt=dt)
    assert d['stable_reduced'] and d['rate_admissible']
    assert np.isclose(d['K'], k, atol=1e-9)
    assert np.isclose(d['rhoGamma'], np.exp(-dt))
    assert np.isclose(d['sigmaGamma'], nndr.optimal_gain(F, 1))
    kc = nndr.critical_K_real(.5) if family == 'real' else nndr.critical_K_complex(2.)
    assert np.isclose(d['Kc_A'], kc)
    assert np.isclose(d['R_A'], k/kc)
    if not np.isclose(d['R_sigma'], 1.):
        assert (d['R_sigma'] > 1.) == (d['sigmaGamma'] > 1.)
    return L, F, d


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def save(fig, out, stem):
    save_figure(fig,out,stem)


def main(out, dt=.5):
    if not np.isfinite(dt) or not .05 <= dt <= 1.45:
        raise ValueError('Use 0.05 <= dt <= 1.45 for this demonstration (2*dt < pi).')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ks = np.linspace(0., 4., 161)
    times = np.linspace(0., 4., 801)
    steps = np.arange(int(np.ceil(4/dt))+1)
    sample_times = steps*dt
    dt_grid = np.linspace(.05, 1.45, 141)
    sweep, response, sampling, summary = [], [], [], {}
    with plt.rc_context():
        configure_style()
        fig, axes = plt.subplots(2,3,figsize=(7.4,5.6))
        fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.89,wspace=.47,hspace=.55)
        fig.legend([plt.Line2D([],[],color=c) for c in COLORS],
                   [f'K = {k:g}' for k in LEVELS],loc='upper center',bbox_to_anchor=(.52,.995),ncol=3)
        for i,family in enumerate(FAMILIES):
            a,b,c = axes[i]
            vals=[]
            for k in ks:
                _,_,d=diagnostics(family,k,dt)
                vals.append((d['R_A'],d['R_sigma']))
                sweep.append((family,k,dt,d['R_A'],d['R_sigma'],d['Kc_A'],d['Kc_sigma'],d['rhoGamma'],d['sigmaGamma']))
            vals=np.asarray(vals)
            _,_,ref=diagnostics(family,1.5,dt)
            ka, ksig=ref['Kc_A'],ref['Kc_sigma']
            lo,hi=sorted((ka,ksig))
            if lo<ks[-1]: a.axvspan(lo,min(hi,ks[-1]),color='#E2E8F0',alpha=.9,label=None)
            a.plot(ks,vals[:,0],color=RA_COLOR,lw=2,label=r'$R_A$')
            a.plot(ks,vals[:,1],color=RS_COLOR,lw=2,ls='--',label=r'$R_\sigma$')
            a.axhline(1,color='.45',ls=':',lw=1)
            a.axvline(1.5,color='.55',ls=':',lw=1)
            a.set(xlabel=r'$K$',ylabel='Ratio',xlim=(0,4))
            a.set_title(TITLES[family]); panel_label(a,chr(97+3*i),x=-.19,y=1.02)
            a.legend(loc='upper left')
            summary[family]={'Kc_A':ka,'Kc_sigma':ksig,'cases':{}}
            for k,col in zip(LEVELS,COLORS):
                L,F,d=diagnostics(family,k,dt)
                G=np.array([nndr.optimal_gain(F,int(j)) for j in steps])
                A=np.array([nndr.variance_response(L,float(t),continuous=True)/2 for t in times])
                assert np.isclose(A[0],1.) and np.isclose(G[0],1.)
                # Independent derivative oracle from the continuous propagator.
                deriv=np.array([np.sum(expm(L*t)*(L@expm(L*t))) for t in times])
                increasing=bool(np.any(deriv>1e-8))
                assert increasing == (d['R_A']>1)
                b.plot(sample_times,G,marker='o',ms=3,lw=1.7,color=col)
                c.plot(times,A,lw=2,color=col)
                if k==LEVELS[-1]:
                    c.plot(times,np.where(deriv>1e-8,A,np.nan),color=col,lw=2.8,solid_capstyle='round')
                response.extend((family,k,'sampled_norm_gain',float(t),float(v)) for t,v in zip(sample_times,G))
                response.extend((family,k,'continuous_variance_normalized',float(t),float(v)) for t,v in zip(times,A))
                summary[family]['cases'][str(k)]={key:float(d[key]) for key in ('R_A','R_sigma','rhoGamma','sigmaGamma')}
                summary[family]['cases'][str(k)].update(increasing_variance_interval=increasing,peak_sampled_gain=float(G.max()),peak_variance_normalized=float(A.max()))
            b.axhline(1,color='.45',ls=':',lw=1);b.set(xlabel=r'$j\Delta t$',ylabel=r'$\|F^j\|_2$',xlim=(0,sample_times[-1]))
            panel_label(b,chr(98+3*i),x=-.19,y=1.02)
            c.axhline(1,color='.45',ls=':',lw=1);c.set(xlabel=r'$t$',ylabel=r'$A(t)/A(0)$',xlim=(0,4))
            panel_label(c,chr(99+3*i),x=-.19,y=1.02)
            for ax in (a,b,c): format_axis(ax,nbx=4,nby=4)
        save(fig,out,'spectral_comparison')
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.0));fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.85,wspace=.36)
        for i,(family,ax) in enumerate(zip(FAMILIES,axes)):
            rows=[]
            for h in dt_grid:
                _,_,d=diagnostics(family,1.5,float(h));rows.append((d['R_A'],d['R_sigma']))
                sampling.append((family,h,1.5,d['R_A'],d['R_sigma']))
            rows=np.asarray(rows)
            ax.plot(dt_grid,rows[:,0],color=RA_COLOR,lw=2,label=r'$R_A$')
            ax.plot(dt_grid,rows[:,1],color=RS_COLOR,lw=2,ls='--',label=r'$R_\sigma$')
            ax.axhline(1,color='.45',ls=':');ax.axvline(dt,color='.55',ls=':',label='Selected interval')
            ax.set(xlabel=r'$\Delta t$',ylabel='Ratio',xlim=(.05,1.45));ax.set_title(TITLES[family]); format_axis(ax); panel_label(ax,chr(97+i),x=-.15,y=1.02); ax.legend()
        save(fig,out,'sampling_effect')
    write_csv(out/'threshold_sweep.csv',['family','K','dt','R_A','R_sigma','Kc_A','Kc_sigma','rho','sigma'],sweep)
    write_csv(out/'response_curves.csv',['family','K','quantity','time','value'],response)
    write_csv(out/'sampling_sweep.csv',['family','dt','K','R_A','R_sigma'],sampling)
    (out/'summary.json').write_text(json.dumps({'dt':dt,'nndr_version':getattr(nndr,'__version__','unknown'),'families':summary},indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,allow_nan=False));print('Saved:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='results/examples/02');p.add_argument('--dt',type=float,default=.5);a=p.parse_args();main(a.out,a.dt)
