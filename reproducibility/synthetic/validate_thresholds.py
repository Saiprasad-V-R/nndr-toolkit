"""Ground-truth validation of revised NNDR thresholds (Appendices B/C).

Run from anywhere after installing nndr and scipy/matplotlib. This script
uses package diagnostics for predictions, matrix exponentials and SVD for
independent numerical ground truth, not the threshold formula for labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nndr.synthetic import real_generator, complex_generator
from nndr.thresholds import threshold_diagnostics

TOL = 1e-8


def reference_kc(kind, a, b):
    # Analytic reference only for choosing K values and measuring ratio error.
    # The response classification below does not use this function.
    if kind == 'real':
        d = (b-a)/(a+b)
        s = 2*np.sqrt(a*b)/(a+b)
        return np.sqrt(s*(1+s))/d
    beta = b/a
    return np.sqrt(np.hypot(1., beta)+1)/beta


def generator(kind, a, b, k):
    return (real_generator if kind == 'real' else complex_generator)(a,b,k)


def normalized_derivative(L, t):
    P = expm(t*L)
    return float(2*np.sum(P*(L@P))/np.sum(P*P))


def oracle(L, kind, a, b):
    # The sign of d(log ||exp(tL)||_F^2)/dt equals the sign of dA/dt.
    # For real pairs, search a log-spaced interval spanning both decay scales.
    # For complex pairs the normalized derivative is periodic (period pi/omega).
    if kind == 'real':
        t = np.r_[0., np.geomspace(1e-6/max(a,b), 12/min(a,b), 180)]
    else:
        t = np.linspace(0., np.pi/b, 181)
    y = np.array([normalized_derivative(L, s) for s in t])
    candidates = [(float(y[0]),float(t[0])), (float(y[-1]),float(t[-1]))]
    # Refine every interior sampled local maximum, not only the global grid max.
    for j in range(1,len(t)-1):
        if y[j] >= y[j-1] and y[j] >= y[j+1] and (y[j]>y[j-1] or y[j]>y[j+1]):
            opt = minimize_scalar(lambda v: -normalized_derivative(L,v),
                                  bounds=(t[j-1],t[j+1]), method='bounded',
                                  options={'xatol':1e-13/max(a,b)})
            candidates.append((float(-opt.fun),float(opt.x)))
    j = int(np.argmax(y))
    candidates.append((float(y[j]),float(t[j])))
    return max(candidates)


def classify(x, tol=TOL):
    if not np.isfinite(x):
        raise ValueError(f'Nonfinite classification value: {x}')
    return 1 if x>tol else -1 if x<-tol else 0


def row(kind,a,b,k,dt,oracle_result=None):
    L = generator(kind,a,b,k)
    F = expm(dt*L)
    d = threshold_diagnostics(F,dt=dt)
    deriv,tmax = oracle_result if oracle_result is not None else oracle(L,kind,a,b)
    sigma = float(np.linalg.svd(F,compute_uv=False)[0])
    ref = k/reference_kc(kind,a,b)
    # Normalize derivative by fastest rate for a dimensionless tolerance.
    truth_A = classify(deriv/max(a,b))
    pred_A = classify(d['R_A']-1)
    truth_sigma = classify(sigma-1)
    pred_sigma = classify(d['R_sigma']-1)
    return dict(kind=kind,rate1=a,rate2_or_omega=b,K=k,dt=dt,
                R_A=d['R_A'],R_A_reference=ref,R_sigma=d['R_sigma'],
                sigma_max=sigma,rho=d['rhoGamma'],max_log_kernel_derivative=deriv,
                time_of_max_log_derivative=tmax,truth_A=truth_A,pred_A=pred_A,
                truth_sigma=truth_sigma,pred_sigma=pred_sigma,
                rate_admissible=d['rate_admissible'])


def sweep():
    rows=[]
    for kind, values in [('real',[1.02,1.1,1.5,2,4,10,100]),
                         ('complex',[.05,.2,.5,1,2,5,10])]:
        for b in values:
            a=1.
            kc=reference_kc(kind,a,b)
            for fraction in [0.,.25,.5,.85,.99,1.,1.01,1.2,2.,4.]:
                k=kc*fraction
                ground=oracle(generator(kind,a,b,k),kind,a,b)
                for dt in np.array([.02,.1,.5])/max(a,b):
                    rows.append(row(kind,a,b,k,float(dt),ground))
    return rows


def make_figure(out):
    fig,axes=plt.subplots(2,3,figsize=(11.2,6.1),layout='constrained')
    for i,(kind,a,b) in enumerate([('real',.5,2.),('complex',1.,2.)]):
        kc=reference_kc(kind,a,b)
        dt=.1
        t=np.linspace(0,4 if kind=='real' else 2.5,400)
        for f in [.65,1.,1.1,1.5]:
            L=generator(kind,a,b,f*kc)
            P=np.array([expm(s*L) for s in t])
            axes[i,0].plot(t,np.sum(P*P,axis=(1,2))/2,label=f'$K/K_c={f:g}$')
        axes[i,0].set(xlabel='Delay $t$',ylabel=r'$\|e^{tL}\|_F^2/2$',title=f'{kind.capitalize()} spectrum: response')
        axes[i,0].legend(fontsize=8,frameon=False)
        xs=np.linspace(0,2,81)
        data=[row(kind,a,b,kc*f,dt) for f in xs]
        axes[i,1].plot([r['R_A'] for r in data],[r['max_log_kernel_derivative'] for r in data])
        axes[i,2].plot([r['R_sigma'] for r in data],[r['sigma_max']-1 for r in data])
        for j,xlabel,ylabel in [(1,r'$R_A$',r'$\max_t\,d\log A(t)/dt$'),(2,r'$R_\sigma$',r'$\sigma_{\max}(F)-1$')]:
            ax=axes[i,j]
            ax.axhline(0,color='.5',lw=.8); ax.axvline(1,color='.5',lw=.8,ls='--')
            ax.set(xlabel=xlabel,ylabel=ylabel,title='Variance response' if j==1 else 'One-step amplification')
        for j in range(3):
            axes[i,j].text(.02,.97,f'({chr(97+3*i+j)})',transform=axes[i,j].transAxes,va='top',fontweight='bold')
    fig.savefig(out/'threshold_ground_truth.png',dpi=220)
    fig.savefig(out/'threshold_ground_truth.svg')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/synthetic/thresholds'))
    p.add_argument('--no-figure',action='store_true')
    args=p.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    rows=sweep()
    with (out/'threshold_sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    stats={}
    for kind in ['real','complex']:
        r=[x for x in rows if x['kind']==kind]
        stats[kind]={'cases':len(r),'A_disagreements':sum(x['truth_A']!=x['pred_A'] for x in r),
                     'sigma_disagreements':sum(x['truth_sigma']!=x['pred_sigma'] for x in r),
                     'A_boundary_cases':sum(x['truth_A']==0 for x in r),
                     'sigma_boundary_cases':sum(x['truth_sigma']==0 for x in r),
                     'A_above':sum(x['truth_A']==1 for x in r),
                     'sigma_above':sum(x['truth_sigma']==1 for x in r),
                     'criteria_differ':sum(x['truth_A']!=x['truth_sigma'] for x in r if x['truth_A']!=0 and x['truth_sigma']!=0),
                     'max_abs_RA_error':max(abs(x['R_A']-x['R_A_reference']) for x in r),
                     'inadmissible':sum(not x['rate_admissible'] for x in r)}
    report={'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},'tolerance':TOL,'results':stats}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert all(s['A_disagreements']==s['sigma_disagreements']==s['inadmissible']==0 and s['max_abs_RA_error']<1e-8 for s in stats.values()), 'Threshold validation failed; inspect CSV.'
    if not args.no_figure: make_figure(out)


if __name__=='__main__':
    main()
