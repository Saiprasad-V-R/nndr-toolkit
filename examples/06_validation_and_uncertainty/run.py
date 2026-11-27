"""Illustrative model checks, fitted normal reference and paired block bootstrap."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import nndr
from nndr.ridge import ridge_fit_F
from nndr.simulate import simulate_var_with_fixed_noise
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _figure_style import configure_style, format_axis, panel_label, save_figure, BLUE, RED, GREEN

LAM = .003
LAG = 10


def fit(X):
    return nndr.fit_relative_operator(X, lam_rel=LAM)


def checks(X):
    F = fit(X)
    m = nndr.method3(F)
    if not m['direction_identifiable']:
        raise ValueError('M3 direction not identifiable')
    P, Y = nndr.adjacent_pairs(X)
    E = m['Q'].T @ (Y-F@P)
    a = nndr.model_adequacy(X, F, m['Q'], max_lag=LAG, lam_rel=LAM)
    d = m['diagnostics']
    return F, a, dict(rho=nndr.spectral_radius(F), sigma=nndr.sigma_max(F),
                      R_A=d['R_A'], R_sigma=d['R_sigma'],
                      prediction=a['prediction'], closure=a['closure_error'],
                      lag_errors=a['lag_errors'],
                      whiteness=nndr.portmanteau_test(E, max_lag=LAG))


def main(out, samples, n_null, n_boot, block, seed):
    if samples < 500 or n_null < 19 or n_boot < 19 or not 1 <= block < samples:
        raise ValueError('Require samples >= 500, null/bootstrap >= 19, and 1 <= block < samples')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # A stable directed 4-channel system with a non-normal 2D block.
    F = np.array([[.8,.9,0,0],[0,.5,0,0],[0,0,.65,0],[0,0,0,.35]])
    X = nndr.simulate_trajectory(F, samples=samples, seed=seed, noise_std=.1)
    # Same physical map, but temporally correlated innovations (unobserved forcing).
    rng = np.random.default_rng(seed+1)
    Z = rng.normal(0,.1,(samples+500,4))
    for k in range(1,len(Z)):
        Z[k] = .75*Z[k-1]+np.sqrt(1-.75**2)*Z[k]
    Xbad = simulate_var_with_fixed_noise(F,Z)[:,500:]
    Fhat,a,good = checks(X)
    _,bad_a,bad = checks(Xbad)
    # Use ambient one-step norm gain: defined even for normal/unstable estimates.
    # Every simulated normal reference is refitted with identical regularization.
    null=[]; normal=None
    for i in range(n_null):
        normal=nndr.variance_matched_surrogate(Fhat,X,seed=seed+100+i)
        null.append(nndr.sigma_max(fit(normal['X'])))
    null=np.asarray(null)
    rank=nndr.surrogate_test(good['sigma'],null)
    P,Y=nndr.adjacent_pairs(X)
    inds=nndr.block_bootstrap(P.shape[1],block,n_resamples=n_boot,seed=seed+3)
    # Resample aligned pairs: no transition across artificial block joins.
    boot=np.array([nndr.sigma_max(ridge_fit_F(P[:,j],Y[:,j],nndr.relative_ridge(P[:,j],LAM))) for j in inds])
    ci=nndr.bootstrap_confidence_interval(good['sigma'],boot,method='percentile')
    phase=nndr.squared_residual_test(X,method='M3',max_lag=LAG,n_surrogates=n_null,lam_rel=LAM,seed=seed+4)
    phase_bad=nndr.squared_residual_test(Xbad,method='M3',max_lag=LAG,n_surrogates=n_null,lam_rel=LAM,seed=seed+5)
    summary=dict(nndr_version=nndr.__version__,samples=samples,seed=seed,lam_rel=LAM,
                 true_rho=nndr.spectral_radius(F),true_sigma=nndr.sigma_max(F),white=good,colored=bad,
                 normal_reference=dict(rho=nndr.spectral_radius(normal['F']),sigma=nndr.sigma_max(normal['F']),
                 target_variance=normal['target_variance'],stationary_variance=float(np.trace(normal['stationary_covariance'])),
                 rank=rank), bootstrap=dict(block=block,n=n_boot,**ci),
                 phase_white={k:v for k,v in phase.items() if k!='null'},
                 phase_colored={k:v for k,v in phase_bad.items() if k!='null'})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.tolist(),allow_nan=False)+'\n')
    for name,values in [('normal_reference',null),('bootstrap_gain',boot),('phase_white',phase['null']),('phase_colored',phase_bad['null'])]:
        np.savetxt(out/(name+'.csv'),values,delimiter=',',header='statistic',comments='')
    with (out/'residual_acf.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['lag','white_r','white_n','colored_r','colored_n'])
        w.writerows(zip(range(1,LAG+1),*a['residual_acf'],*bad_a['residual_acf']))
    configure_style()
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.8))
    for ax,lab,vals,obs,xlabel in [(axs[0,0],'a',null,good['sigma'],r'$\sigma_{\max}(\widehat F)$'),(axs[0,1],'b',boot,good['sigma'],r'$\sigma_{\max}(\widehat F)$')]:
        ax.hist(vals,bins=18,color=BLUE,alpha=.7,edgecolor='white',label='Normal reference' if lab=='a' else 'Block bootstrap')
        ax.axvline(obs,color=RED,label='Observed')
        if lab=='b':
            for j,k in enumerate(('lower','upper')): ax.axvline(ci[k],color='0.25',ls='--',label='95% interval' if j==0 else None)
        ax.set(xlabel=xlabel,ylabel='Count');ax.legend(loc='best');format_axis(ax);panel_label(ax,lab,x=-.16,y=1.02)
    for ax,lab,diag,col,title in [(axs[1,0],'c',a,BLUE,'White innovations'),(axs[1,1],'d',bad_a,RED,'Colored innovations')]:
        l=np.arange(1,LAG+1)
        ax.axhline(0,color='0.5',lw=.8)
        ax.plot(l,diag['residual_acf'][0],color=col,marker='o',ms=3,label=r'$e_r$')
        ax.plot(l,diag['residual_acf'][1],color=col,ls='--',marker='s',ms=3,label=r'$e_n$')
        ax.set(xlabel='Lag (samples)',ylabel='Residual autocorrelation',title=title);ax.legend();format_axis(ax);ax.xaxis.set_major_locator(MaxNLocator(integer=True));panel_label(ax,lab,x=-.16,y=1.02)
    # Same limits for the residual comparisons, with data-driven range.
    v=np.concatenate([a['residual_acf'].ravel(),bad_a['residual_acf'].ravel(),[0]])
    pad=max(.01,.08*np.ptp(v))
    for ax in axs[1]:ax.set_ylim(v.min()-pad,v.max()+pad)
    fig.subplots_adjust(left=.12,right=.97,bottom=.12,top=.92,wspace=.36,hspace=.52)
    save_figure(fig,out,'validation_and_uncertainty')
    print(json.dumps(summary,indent=2,default=lambda x:x.tolist(),allow_nan=False));print('Saved:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='results/example06');p.add_argument('--samples',type=int,default=3000);p.add_argument('--n-null',type=int,default=99);p.add_argument('--n-bootstrap',type=int,default=199);p.add_argument('--block-length',type=int,default=30);p.add_argument('--seed',type=int,default=2);a=p.parse_args();main(a.out,a.samples,a.n_null,a.n_bootstrap,a.block_length,a.seed)
