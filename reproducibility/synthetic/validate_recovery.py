"""Reproduce the controlled noisy high-dimensional NNDR recovery benchmark.

Uses installed NNDR methods, ridge and threshold kernel; no threshold clipping
or stabilization. This is a new demonstration, not a reproduction of an
original numbered paper figure.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nndr.synthetic import real_generator, complex_generator
from nndr.synthetic_recovery import embedded_map, simulate_trajectory
from nndr.methods import method1_eigenbasis_svd, method2_optimization, method3_commutator
from nndr.ridge import ridge_fit_F
from nndr.thresholds import threshold_diagnostics

METHODS = {'M1': method1_eigenbasis_svd, 'M2': method2_optimization, 'M3': method3_commutator}
DT = .1
LENGTHS = (200, 1000, 5000)
TARGETS = (.25, .65, 1.35, 2.)
N = 12
T_TEST = 2000
NOISE_STD = .1
RIDGE_REL = 1e-6


def canonical(family, target):
    # Set target RA using independent analytic thresholds for the known rates.
    if family == 'real':
        s = .8  # gamma1=.5, gamma2=2 -> Delta=.6
        K = target*np.sqrt(s/(1-s))
        return real_generator(.5, 2., K)
    K = target/np.sqrt(np.hypot(1., 2.)-1.)
    return complex_generator(1., 2., K)


def largest_angle(Q, P):
    return float(np.degrees(np.arccos(np.clip(np.linalg.svd(Q.T @ P, compute_uv=False)[-1], 0, 1))))


def observable_corr(Q, X):
    # Independent reference: least-squares projection of centered mean field
    # onto centered plane coordinates, not the existing covariance heuristic.
    Z = (Q.T @ X).T
    Z -= Z.mean(axis=0)
    y = X.mean(axis=0)
    y -= y.mean()
    norm = np.linalg.norm(y)
    if norm == 0:
        return np.nan
    a = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.linalg.norm(Z @ a)/norm)


def metrics(F, Q, F_true, Q_true, Xtest, truth, true_corr):
    d = threshold_diagnostics(Q.T @ F @ Q, dt=DT)
    r = Q[:, 0]
    ra, rs = d['R_A'], d['R_sigma']
    # Support uses existing core Q[:,0] convention. It is basis dependent.
    same_family = d['spectrum_type'] == truth['spectrum_type']
    split = d['rate_splitting']
    rho = float(max(abs(np.linalg.eigvals(F))))
    sg = float(np.linalg.svd(F, compute_uv=False)[0])
    c = observable_corr(Q, Xtest)
    return dict(R_A=ra, R_sigma=rs, K=d['K'], splitting=split,
                spectrum_type=d['spectrum_type'], spectrum_match=int(same_family),
                rate_status=d['rate_status'], stable_full=int(rho<1), stable_reduced=int(d['stable_reduced']),
                rho_full=rho, rho_reduced=d['rhoGamma'], sigma_full=sg, sigma_reduced=d['sigmaGamma'],
                angle_deg=largest_angle(Q, Q_true), support=float(np.linalg.norm(r,1)/(np.sqrt(r.size)*np.linalg.norm(r))),
                meanfield_corr=c, meanfield_corr_abs_error=abs(c-true_corr),
                capture_fraction=d['sigmaGamma']/sg,
                operator_rel_error=float(np.linalg.norm(F-F_true,'fro')/np.linalg.norm(F_true,'fro')),
                RA_abs_error=abs(ra-truth['R_A']), Rsigma_abs_error=abs(rs-truth['R_sigma']),
                K_abs_error=abs(d['K']-truth['K']),
                splitting_abs_error=abs(split-truth['rate_splitting']) if same_family else np.nan,
                RA_valid=int(np.isfinite(ra)), Rsigma_valid=int(np.isfinite(rs)),
                RA_correct=int(np.isfinite(ra) and ((ra>1)==(truth['R_A']>1))),
                Rsigma_correct=int(np.isfinite(rs) and ((rs>1)==(truth['sigmaGamma']>1))))


def run(reps):
    rows=[]
    for fi,family in enumerate(('real','complex')):
        for ti,target in enumerate(TARGETS):
            L=canonical(family,target)
            for rep in range(reps):
                seed=100000*fi+10000*ti+rep
                F,Q=embedded_map(L, dimension=N, dt=DT, seed=seed)
                truth=threshold_diagnostics(Q.T@F@Q,dt=DT)
                assert np.isclose(truth['R_A'],target,rtol=1e-10)
                X=simulate_trajectory(F,samples=max(LENGTHS),noise_std=NOISE_STD,seed=seed+500000)
                Xtest=simulate_trajectory(F,samples=T_TEST,noise_std=NOISE_STD,seed=seed+1000000)
                c0=observable_corr(Q,Xtest)
                common=dict(family=family,target_RA=target,rep=rep,seed=seed,truth_RA=truth['R_A'],truth_Rsigma=truth['R_sigma'],truth_K=truth['K'],truth_splitting=truth['rate_splitting'],truth_meanfield_corr=c0)
                rows.append(dict(common,stage='truth',method='Q_true',samples=0,**metrics(F,Q,F,Q,Xtest,truth,c0)))
                for m,fn in METHODS.items():
                    p=fn(F,seed=0)['Q'] if m=='M2' else fn(F)['Q']
                    rows.append(dict(common,stage='oracle',method=m,samples=0,**metrics(F,p,F,Q,Xtest,truth,c0)))
                for T in LENGTHS:
                    X0,Y=X[:,:T-1],X[:,1:T]
                    lam=RIDGE_REL*np.sum(X0*X0)/N
                    Fhat=ridge_fit_F(X0,Y,lam)
                    rows.append(dict(common,stage='known_plane',method='Q_true',samples=T,**metrics(Fhat,Q,F,Q,Xtest,truth,c0)))
                    for m,fn in METHODS.items():
                        p=fn(Fhat,seed=0)['Q'] if m=='M2' else fn(Fhat)['Q']
                        rows.append(dict(common,stage='estimated',method=m,samples=T,**metrics(Fhat,p,F,Q,Xtest,truth,c0)))
    return rows


def quantile(rows,key,q):
    v=np.array([r[key] for r in rows],float)
    v=v[np.isfinite(v)]
    return float(np.quantile(v,q)) if v.size else None


def summarize(rows):
    out=[]
    groups=sorted({(r['family'],r['stage'],r['method'],r['samples']) for r in rows})
    for family,stage,m,T in groups:
        rr=[r for r in rows if (r['family'],r['stage'],r['method'],r['samples'])==(family,stage,m,T)]
        d=dict(family=family,stage=stage,method=m,samples=T,count=len(rr))
        for k in ('angle_deg','RA_abs_error','Rsigma_abs_error','K_abs_error','splitting_abs_error','meanfield_corr_abs_error','operator_rel_error','capture_fraction','support'):
            for q,suffix in [(0.25,'q25'),(.5,'median'),(.75,'q75')]:
                d[k+'_'+suffix]=quantile(rr,k,q)
        for k in ('RA_valid','Rsigma_valid','RA_correct','Rsigma_correct','stable_full','stable_reduced','spectrum_match'):
            d[k+'_fraction']=float(np.mean([r[k] for r in rr]))
        out.append(d)
    return out


def plot(summary,out):
    fig,axes=plt.subplots(2,3,figsize=(11,6.4),constrained_layout=True)
    for i,f in enumerate(('real','complex')):
        for j,(metric,ylabel) in enumerate([('angle_deg','Largest principal angle (deg)'),('RA_abs_error',r'$|\widehat{R}_A-R_A|$'),('Rsigma_abs_error',r'$|\widehat{R}_\sigma-R_\sigma|$')]):
            ax=axes[i,j]
            for m in METHODS:
                r=[r for r in summary if r['family']==f and r['stage']=='estimated' and r['method']==m]
                r=sorted(r,key=lambda r:r['samples'])
                x=np.array([r['samples'] for r in r]); y=np.array([r[metric+'_median'] for r in r],float)
                lo=np.array([r[metric+'_q25'] for r in r],float); hi=np.array([r[metric+'_q75'] for r in r],float)
                line,=ax.plot(x,y,'o-',label=m);ax.fill_between(x,lo,hi,alpha=.15,color=line.get_color())
            ax.set_xscale('log');ax.set_xticks(LENGTHS,labels=[str(t) for t in LENGTHS]);ax.set_ylabel(ylabel)
            ax.set_title(('Real' if i==0 else 'Complex')+' spectrum · '+chr(97+i*3+j))
            if i==1:ax.set_xlabel('Observed training samples')
            if i==0 and j==0:ax.legend(frameon=False)
            ax.grid(alpha=.2)
    fig.savefig(out/'recovery_benchmark.png',dpi=220);fig.savefig(out/'recovery_benchmark.svg');plt.close(fig)


def write_csv(path, rows):
    with path.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--reps',type=int,default=12);a=p.parse_args()
    if a.reps<1:p.error('--reps must be positive')
    a.out.mkdir(parents=True,exist_ok=True)
    rows=run(a.reps);summary=summarize(rows)
    write_csv(a.out/'recovery_trials.csv',rows);write_csv(a.out/'recovery_summary.csv',summary);plot(summary,a.out)
    config=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),dimension=N,dt=DT,training_lengths=LENGTHS,RA_targets=TARGETS,replicates=a.reps,test_samples=T_TEST,noise_std=NOISE_STD,relative_ridge=RIDGE_REL,rows=len(rows),summary=summary)
    (a.out/'recovery_summary.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    for r in summary:
        if r['stage']=='estimated' and r['samples']==max(LENGTHS):
            print(r['family'],r['method'],'angle=',r['angle_deg_median'],'RA median absolute error=',r['RA_abs_error_median'],'RA correct=',r['RA_correct_fraction'],'Rsigma correct=',r['Rsigma_correct_fraction'])
    print('Trial rows:',len(rows),'Output:',a.out)

if __name__=='__main__':main()
