#!/usr/bin/env python3
"""Example 03: infer a directed network's hidden response plane from observations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nndr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _figure_style import configure_style, format_axis, panel_label, save_figure, BLUE, RED, GREEN

COLORS = {'M1':BLUE, 'M2':RED, 'M3':GREEN}


def build_network():
    """Two weighted groups with one-way coupling and a planted invariant plane."""
    a = np.r_[np.array([1, 2, 3, 3, 2, 1], float), np.zeros(6)]
    b = np.r_[np.zeros(6), np.array([1, 2, 3, 2, 1, 1], float)]
    Q = np.column_stack([a/np.linalg.norm(a), b/np.linalg.norm(b)])
    block = np.array([[.8, .9], [0., .5]])
    F = .25*np.eye(12) + Q @ (block-.25*np.eye(2)) @ Q.T
    return F, Q


def csv_file(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main(out, samples=5000, seed=2):
    if samples < 100:
        raise ValueError('Use at least 100 samples for this demonstration.')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    F, Q = build_network()
    dt = 1.0
    assert np.allclose(F @ Q, Q @ nndr.project_operator(F, Q))
    assert nndr.spectral_radius(F) < 1
    # The reference ordering is defined by maximum directed coupling, not group membership.
    Qref, _, identifiable = nndr.order_response_plane(F, Q)
    assert identifiable
    truth = nndr.threshold_diagnostics(nndr.project_operator(F, Q), dt=dt)
    X = nndr.simulate_trajectory(F, samples=samples, noise_std=.1, burn_in=500, seed=seed)
    # Only X is passed to estimation. No true operator or plane enters this step.
    Fhat, info = nndr.fit_relative_operator(X, lam_rel=.003, shrinkage=0., return_info=True)
    results = {name: func(Fhat, dt=dt) for name, func in
               [('M1',nndr.method1),('M2',nndr.method2),('M3',nndr.method3)]}
    rows = []
    for name, v in results.items():
        d = v['diagnostics']
        assert np.allclose(v['Q'].T @ v['Q'], np.eye(2))
        assert np.allclose(v['Gamma'], nndr.project_operator(Fhat, v['Q']))
        direction_ok = v.get('direction_identifiable')
        rows.append(dict(method=name, angle_deg=nndr.principal_angle(v['Q'],Q),
            R_A=float(d['R_A']), R_sigma=float(d['R_sigma']),
            reduced_stable=bool(d['stable_reduced']), rate_status=d['rate_status'],
            reaction_overlap=float(abs(v['r'] @ Qref[:,0])) if direction_ok is not False else None,
            input_overlap=float(abs(v['n'] @ Qref[:,1])) if direction_ok is not False else None,
            support=nndr.reaction_support(v['r']) if direction_ok is not False else None,
            capture_fraction=nndr.capture_fraction(Fhat,v['Q']),
            mean_field_alignment=nndr.plane_correlation(X,v['Q']),
            plane_identifiable=v.get('plane_identifiable'), direction_identifiable=direction_ok))
    summary = dict(nndr_version=nndr.__version__, samples=samples, seed=seed, dt=dt,
        innovation_sd=.1, burn_in=500, fit=info, shrinkage=0.,
        spectral_radius_true=nndr.spectral_radius(F), spectral_radius_fit=nndr.spectral_radius(Fhat),
        relative_operator_error=nndr.relative_operator_error(Fhat,F),
        truth=dict(R_A=float(truth['R_A']), R_sigma=float(truth['R_sigma']),
                   support=nndr.reaction_support(Qref[:,0])),
        M2_M3_angle_deg=nndr.principal_angle(results['M2']['Q'],results['M3']['Q']), methods=rows)
    # JSON null represents inadmissible nonfinite values; CSV retains the numeric NaNs.
    def clean(v):
        if isinstance(v,dict): return {k:clean(w) for k,w in v.items()}
        if isinstance(v,list): return [clean(w) for w in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
        if isinstance(v,np.generic): return v.item()
        return v
    (out/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    csv_file(out/'method_diagnostics.csv',list(rows[0]),(list(row.values()) for row in rows))
    nodes = np.arange(1,13)
    for name,M in [('true_operator',F),('estimated_operator',Fhat)]:
        csv_file(out/(name+'.csv'),['target_node']+[f'source_{i}' for i in nodes],
                 ([i,*M[i-1]] for i in nodes))
    csv_file(out/'observations.csv',['time']+[f'node_{i}' for i in nodes],
             ([i*dt,*X[:,i]] for i in range(samples)))
    csv_file(out/'direction_weights.csv',['node','reference_r2','reference_n2']+
             [f'{m}_{q}2' for m in results for q in ('r','n')],
             ([i,Qref[i-1,0]**2,Qref[i-1,1]**2]+[
               v[q][i-1]**2 if v.get('direction_identifiable') is not False else np.nan
               for v in results.values() for q in ('r','n')] for i in nodes))

    with plt.rc_context():
        configure_style()
        fig, axes = plt.subplots(2,3,figsize=(7.5,5.8))
        fig.subplots_adjust(left=.10,right=.96,bottom=.15,top=.92,wspace=.52,hspace=.50)
        a,b,c,d,e,f = axes.flat
        # Only draw the strongest non-self edges. The matrix panel shows all weights.
        theta = np.linspace(0,2*np.pi,6,endpoint=False)+np.pi/6
        xy = np.vstack([np.c_[-.72+.40*np.cos(theta),.70*np.sin(theta)],
                        np.c_[.72+.40*np.cos(theta),.70*np.sin(theta)]])
        threshold=.16
        for i,j in zip(*np.where(F>=threshold)):
            if i==j: continue
            a.annotate('',xy=xy[i],xytext=xy[j],arrowprops=dict(arrowstyle='-|>',
                color='#668D90',alpha=.42,lw=.6+2*F[i,j],shrinkA=11,shrinkB=11,
                connectionstyle='arc3,rad=.08'))
        for i,p in enumerate(xy):
            a.scatter(*p,s=90,c='#D4E8E7' if i<6 else '#E8EDF4',edgecolor='#315B65' if i<6 else '#61758B',zorder=3)
            a.text(*p,str(i+1),ha='center',va='center',fontsize=7,zorder=4)
        a.text(-.72,-.98,'Group A',ha='center',color=BLUE)
        a.text(.72,-.98,'Group B',ha='center',color='.35')
        a.set(xlim=(-1.35,1.35),ylim=(-1.12,.94));a.set_axis_off()
        a.set_title('Network')
        # Unscaled, offset traces of representative nodes; common amplitude units.
        length=min(150,samples); selected=[2,4,7,8]; sep=max(.5,2.4*np.max(np.abs(X[selected,:length])))
        for j,i in enumerate(selected):
            b.plot(np.arange(length),X[i,:length]+j*sep,lw=1.15,color=BLUE if i<6 else '#666666')
        b.set(yticks=np.arange(len(selected))*sep,yticklabels=[f'Node {i+1}' for i in selected],xlabel='Sample index')
        b.set_title('Observations')
        lim=max(abs(F).max(),abs(Fhat).max());im=c.imshow(Fhat,vmin=-lim,vmax=lim,cmap='RdBu_r',interpolation='nearest')
        c.axhline(5.5,color='#44515D',lw=.75);c.axvline(5.5,color='#44515D',lw=.75)
        c.set(xticks=[0,5,6,11],xticklabels=[1,6,7,12],yticks=[0,5,6,11],yticklabels=[1,6,7,12],xlabel='Source node',ylabel='Target node')
        c.set_title('Estimated map');fig.colorbar(im,ax=c,fraction=.045,pad=.04)
        for ax,q,col,title in [(d,'r',0,'Reaction direction'),(e,'n',1,'Input direction')]:
            ax.bar(nodes,Qref[:,col]**2,color='#DAE2E7',width=.78,label='Reference',zorder=1)
            for m,v in results.items():
                if v.get('direction_identifiable') is False: continue
                ax.plot(nodes,v[q]**2,'-o',lw=1.35,ms=3,color=COLORS[m],label=m,zorder=3)
            ax.axvline(6.5,color='#9BA7AF',ls=':',lw=1)
            ax.set(xticks=[1,3,6,7,9,12],xlabel='Node',ylabel=r'$|'+q+r'_i|^2$',xlim=(.5,12.5),ylim=(0,None));ax.set_title(title)
        angles=[row['angle_deg'] for row in rows];bars=f.bar(list(results),angles,color=[COLORS[m] for m in results],width=.6)
        f.set(ylabel=r'Principal angle ($^\circ$)',ylim=(0,max(angles)*1.18+1)); f.set_title('Plane error')
        for ax,label in zip(axes.flat,'abcdef'): panel_label(ax,label,x=-.17,y=1.02)
        # Preserve explicit node and categorical ticks; only format continuous axes.
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator
        b.xaxis.set_major_locator(MaxNLocator(4)); b.xaxis.set_minor_locator(AutoMinorLocator())
        for ax in (d,e,f):
            ax.yaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_minor_locator(AutoMinorLocator())
        fig.legend([Line2D([0],[0],color='#DAE2E7',lw=7)]+[Line2D([0],[0],color=COLORS[m],lw=2,marker='o') for m in results],['Reference','M1','M2','M3'],ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.5,.025))
        save_figure(fig,out,'network_inference')
        fig,axs=plt.subplots(1,2,figsize=(6.6,3.2));fig.subplots_adjust(left=.12,right=.86,top=.86,bottom=.20,wspace=.38)
        for ax,M,title in zip(axs,[F,Fhat],['True map','Estimated map']):
            im=ax.imshow(M,vmin=-lim,vmax=lim,cmap='RdBu_r',interpolation='nearest');ax.axvline(5.5,color='#44515D',lw=.75);ax.axhline(5.5,color='#44515D',lw=.75)
            ax.set(title=title,xticks=[0,5,6,11],xticklabels=[1,6,7,12],yticks=[0,5,6,11],yticklabels=[1,6,7,12],xlabel='Source node',ylabel='Target node')
        for ax,label in zip(axs,'ab'): panel_label(ax,label,x=-.15,y=1.02)
        fig.colorbar(im,cax=fig.add_axes([.90,.23,.022,.59]),label='Map coefficient')
        save_figure(fig,out,'operator_comparison')
    print(json.dumps(clean(summary),indent=2,allow_nan=False));print('Saved outputs to',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('results/examples/03_network'));p.add_argument('--samples',type=int,default=5000);p.add_argument('--seed',type=int,default=2);a=p.parse_args();main(a.out,a.samples,a.seed)
