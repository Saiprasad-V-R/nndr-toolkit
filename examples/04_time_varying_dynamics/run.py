#!/usr/bin/env python3
"""Track non-normal amplification in a stable time-varying stochastic network."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nndr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _figure_style import configure_style, format_axis, panel_label, save_figure, BLUE, RED


def coupling(t, samples):
    """Smooth coupling pulse; time is measured in sampling steps."""
    return .25 + .80*np.exp(-.5*((np.asarray(t)-.5*samples)/(.09*samples))**2)


def make_model():
    a = np.r_[1., 2., 3., 4., 0., 0., 0., 0.]
    b = np.r_[0., 0., 0., 0., 4., 3., 2., 1.]
    Q = np.column_stack([a/np.linalg.norm(a), b/np.linalg.norm(b)])
    base = .25*np.eye(8) + Q @ np.diag([.55,.25]) @ Q.T
    cross = np.outer(Q[:,0], Q[:,1])
    return base, cross, Q


def csv_write(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(fields); w.writerows(rows)


def clean(v):
    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v, list): return [clean(x) for x in v]
    if isinstance(v, (float,np.floating)) and not np.isfinite(v): return None
    if isinstance(v, np.generic): return v.item()
    return v


def main(out, samples=12000, short=600, long=1800, step=100, seed=4):
    if not 100 <= short < long <= samples or step < 1:
        raise ValueError('Require 100 <= short < long <= samples and positive step.')
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    base, cross, Q = make_model()
    times=np.arange(samples+1)
    c=coupling(times,samples)
    # The exact local map is base+c[k]*cross for transition k -> k+1.
    # Burn in at the initial map; keep only observed samples.
    rng=np.random.default_rng(seed)
    x=np.zeros(8)
    F0=base+c[0]*cross
    for eps in rng.normal(0,.1,(500,8)):
        x=F0@x+eps
    X=np.empty((8,samples)); X[:,0]=x
    for k,eps in enumerate(rng.normal(0,.1,(samples-1,8))):
        x=(base+c[k]*cross)@x+eps
        X[:,k+1]=x
    # Analytic model structure fixes rho=0.8 and the invariant plane for every k.
    # Compute reference diagnostics through the public NNDR API.
    # Dense plotting grid includes the peak, endpoints, and all inference times.
    ends=np.arange(long,samples+1,step)
    grid=np.unique(np.r_[np.arange(0,samples+1,max(1,samples//400)),ends,samples,samples//2])
    truth=[]
    for k in grid:
        F=base+c[k]*cross
        assert np.allclose(F@Q,Q@nndr.project_operator(F,Q))
        d=nndr.threshold_diagnostics(nndr.project_operator(F,Q),dt=1.)
        rho=nndr.spectral_radius(F)
        assert np.isclose(rho,.8)
        truth.append(dict(time=int(k),coupling=float(c[k]),rho=rho,R_A=float(d['R_A']),R_sigma=float(d['R_sigma'])))
    ref={r['time']:r for r in truth}
    rows=[]
    for w in (short,long):
        for e in ends:
            # Only samples [e-w,e) enter the fit. No local truth enters estimation.
            Fhat,info=nndr.fit_relative_operator(X[:,e-w:e],lam_rel=.003,shrinkage=0.,return_info=True)
            m=nndr.method3(Fhat,dt=1.)
            d=m['diagnostics']; identifiable=m['plane_identifiable']; direction_ok=m['direction_identifiable']
            rows.append(dict(window=w,t_start=int(e-w),t_end=int(e),last_sample=int(e-1),
                rho=nndr.spectral_radius(Fhat),R_A=float(d['R_A']),R_sigma=float(d['R_sigma']),
                stable_full=bool(nndr.spectral_radius(Fhat)<1),stable_reduced=bool(d['stable_reduced']),
                rate_status=d['rate_status'],plane_identifiable=bool(identifiable),direction_identifiable=bool(direction_ok),
                plane_angle_deg=nndr.principal_angle(m['Q'],Q) if identifiable else np.nan,
                support=nndr.reaction_support(m['r']) if direction_ok else np.nan,
                condition=info['condition'],truth_R_A=ref[e]['R_A'],truth_R_sigma=ref[e]['R_sigma']))
    def values(records,key): return np.asarray([r[key] for r in records],float)
    summary=dict(nndr_version=nndr.__version__,samples=samples,seed=seed,innovation_sd=.1,burn_in=500,
                 lam_rel=.003,shrinkage=0.,dt=1.,window_step=step,true_rho=.8,windows=[])
    for w in (short,long):
        a=[r for r in rows if r['window']==w]
        entry=dict(window=w,count=len(a),stable_full_count=sum(r['stable_full'] for r in a),
                   stable_reduced_count=sum(r['stable_reduced'] for r in a),
                   identifiable_plane_count=sum(r['plane_identifiable'] for r in a))
        for key in ('R_A','R_sigma','plane_angle_deg'):
            v=values(a,key); valid=np.isfinite(v)
            entry[key+'_finite_count']=int(valid.sum())
            entry[key+'_median']=float(np.median(v[valid])) if valid.any() else None
            if key!='plane_angle_deg':
                true=values(a,'truth_'+key)
                entry[key+'_rmse_to_endpoint_truth']=float(np.sqrt(np.mean((v[valid]-true[valid])**2))) if valid.any() else None
                entry[key+'_peak']=float(np.max(v[valid])) if valid.any() else None
                entry[key+'_peak_endpoint']=int(values(a,'t_end')[np.flatnonzero(valid)[np.argmax(v[valid])]]) if valid.any() else None
        summary['windows'].append(entry)
    csv_write(out/'observations.csv',['sample']+[f'node_{i}' for i in range(1,9)],([k,*X[:,k]] for k in range(samples)))
    csv_write(out/'local_truth.csv',list(truth[0]),(list(r.values()) for r in truth))
    csv_write(out/'window_diagnostics.csv',list(rows[0]),(list(r.values()) for r in rows))
    (out/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with plt.rc_context():
        configure_style()
        fig,ax=plt.subplots(4,1,sharex=True,figsize=(6.5,6.3))
        fig.subplots_adjust(left=.15,right=.97,top=.95,bottom=.09,hspace=.18)
        t=values(truth,'time')
        ax[0].plot(t,values(truth,'coupling'),color='0.2')
        ax[0].set_ylabel(r'$c_k$')
        for a,key,label in zip(ax[1:],('rho','R_A','R_sigma'),(r'$\rho(F)$',r'$R_A$',r'$R_\sigma$')):
            a.plot(t,values(truth,key),color='0.2',linestyle='--',label='Local truth',zorder=4)
            for w,color,ls in [(short,BLUE,'-'),(long,RED,'-.')]:
                rs=[r for r in rows if r['window']==w]
                a.plot(values(rs,'t_end'),values(rs,key),color=color,linestyle=ls,label=f'{w} samples')
            a.axhline(1.,color='0.6',linestyle=':',linewidth=.9,zorder=0)
            a.set_ylabel(label)
        ax[1].legend(loc='upper center',ncol=3)
        for i,a in enumerate(ax):
            format_axis(a,nbx=6,nby=4); panel_label(a,chr(97+i),x=-.12,y=1.02)
            a.set_xlim(0,samples)
        ax[-1].set_xlabel('Exclusive window endpoint (sampling steps)')
        save_figure(fig,out,'time_varying_dynamics')
    print(json.dumps(clean(summary),indent=2,allow_nan=False))
    print('Saved:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='results/examples/04')
    p.add_argument('--samples',type=int,default=12000)
    p.add_argument('--window-short',type=int,default=600)
    p.add_argument('--window-long',type=int,default=1800)
    p.add_argument('--step',type=int,default=100)
    p.add_argument('--seed',type=int,default=4)
    a=p.parse_args(); main(a.out,a.samples,a.window_short,a.window_long,a.step,a.seed)
