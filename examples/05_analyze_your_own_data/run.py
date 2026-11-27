#!/usr/bin/env python3
"""Analyze a regularly sampled numeric CSV using trailing-window NNDR."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nndr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _figure_style import configure_style, format_axis, panel_label, save_figure, BLUE, RED, GREEN, C_PUR

HERE=Path(__file__).resolve().parent


def load_recording(path, time_column='time', channels=None, dt=None):
    """Return time, channel-by-sample array, selected names, sampling interval."""
    if dt is not None and (not np.isfinite(dt) or dt<=0):
        raise ValueError('dt must be finite and positive')
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.reader(f)
        header=next(reader,[])
        if not header or len(set(header))!=len(header) or any(not h.strip() for h in header):
            raise ValueError('CSV requires unique, nonempty column names')
        for line,row in enumerate(reader,2):
            if len(row)!=len(header):
                raise ValueError(f'CSV row {line} has wrong number of fields')
    df=pd.read_csv(path,encoding='utf-8-sig')
    if time_column is not None and time_column not in df:
        raise ValueError('Missing timestamp column; use --time-column or --no-time-column --dt SECONDS')
    names=list(channels) if channels is not None else [c for c in header if c!=time_column]
    if len(names)<2 or len(set(names))!=len(names) or time_column in names or any(c not in df for c in names):
        raise ValueError('Select at least two distinct existing measurement columns, excluding time')
    X=df[names].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float).T
    if X.shape[1]<3 or not np.isfinite(X).all():
        raise ValueError('Require at least three samples and finite measurements; no automatic imputation')
    if time_column is None:
        if dt is None: raise ValueError('--no-time-column requires --dt in seconds')
        t=np.arange(X.shape[1])*dt
    else:
        t=pd.to_numeric(df[time_column],errors='raise').to_numpy(dtype=float)
        d=np.diff(t)
        if not np.isfinite(t).all() or np.any(d<=0):
            raise ValueError('Timestamps must be finite, strictly increasing numeric seconds')
        inferred=float(np.median(d))
        if not np.allclose(d,inferred,rtol=1e-5,atol=inferred*1e-8):
            raise ValueError('Irregular sampling: establish a justified regular grid first')
        if dt is not None and not np.isclose(dt,inferred,rtol=1e-5,atol=inferred*1e-8):
            raise ValueError('Supplied dt disagrees with timestamps')
        dt=inferred if dt is None else float(dt)
    return t,X,names,float(dt)


def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def write_csv(path,rows,columns):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def analyze_windows(t,X,names,dt,window,step,lam_rel,shrinkage,centering,methods):
    if not max(3,len(names)+1)<=window<=X.shape[1] or step<1:
        raise ValueError('Require window >= channels+1 and >=3, window <= samples, step >=1')
    if not np.isfinite(lam_rel) or lam_rel<0 or not np.isfinite(shrinkage) or not 0<=shrinkage<1:
        raise ValueError('Require nonnegative finite relative ridge and 0 <= shrinkage < 1')
    rows=[]; directions=[]; last=X.shape[1] - (X.shape[1]-window)%step
    for e in range(window,X.shape[1]+1,step):
        Z=X[:,e-window:e].copy()
        if centering=='window': Z-=Z.mean(axis=1,keepdims=True)
        base=dict(start_index=e-window,end_index=e,last_sample_time_s=float(t[e-1]),
                  endpoint_s=float(t[e-1]+dt))
        try:
            F,info=nndr.fit_relative_operator(Z,lam_rel=lam_rel,shrinkage=shrinkage,return_info=True)
            rho=nndr.spectral_radius(F)
        except (ValueError,np.linalg.LinAlgError) as exc:
            for method in methods: rows.append(dict(**base,method=method,status='fit_failed',error=str(exc)))
            continue
        results={}; current=[]
        for method in methods:
            row=dict(**base,method=method,status='ok',error='',rho_F=rho,stable_full=bool(rho<1),gram_condition=info['condition'])
            try:
                m=getattr(nndr,'method'+method[1])(F,dt=dt)
                d=m['diagnostics']; plane_ok=m.get('plane_identifiable'); direction_ok=m.get('direction_identifiable')
                # M3 returns an explicit flag; M2 does not certify uniqueness.
                row.update(R_A=d['R_A'],R_sigma=d['R_sigma'],rho_Gamma=d['rhoGamma'],
                           stable_reduced=d['stable_reduced'],rate_status=d['rate_status'],
                           plane_identifiable=plane_ok,direction_identifiable=direction_ok,
                           support=nndr.reaction_support(m['r']) if direction_ok is not False else np.nan,
                           capture_fraction=nndr.capture_fraction(F,m['Q']) if plane_ok is not False else np.nan)
                if plane_ok is False:
                    row.update(status='plane_unidentifiable',R_A=np.nan,R_sigma=np.nan)
                else:
                    results[method]=m
                if e==last and plane_ok is not False and direction_ok is not False:
                    for i,name in enumerate(names):
                        directions.append(dict(method=method,channel=name,endpoint_s=base['endpoint_s'],
                                               reaction_weight=float(abs(m['r'][i])),input_weight=float(abs(m['n'][i]))))
            except (ValueError,np.linalg.LinAlgError) as exc:
                row=dict(**base,method=method,status='extraction_failed',error=str(exc),
                         rho_F=rho,stable_full=bool(rho<1),gram_condition=info['condition'])
            current.append(row)
        angle=nndr.principal_angle(results['M2']['Q'],results['M3']['Q']) if set(results)=={'M2','M3'} else np.nan
        for row in current: row['M2_M3_angle_deg']=angle
        rows.extend(current)
    return rows,directions,last


def plot(out,t,X,names,rows,directions,last,methods,dt):
    colors={'M2':BLUE,'M3':RED}; styles={'M2':'-','M3':'--'}
    with plt.rc_context():
        configure_style()
        fig,ax=plt.subplots(4,1,sharex=True,figsize=(6.5,6.3))
        fig.subplots_adjust(left=.15,right=.97,top=.96,bottom=.09,hspace=.18)
        palette=[BLUE,RED,GREEN,C_PUR]
        for i,name in enumerate(names):
            ax[0].plot(t,X[i],color=palette[i%4],ls=('-', '--', '-.', ':')[(i//4)%4],lw=.65,alpha=.85,label=name)
        ax[0].set_ylabel('Signal')
        if len(names)<=8: ax[0].legend(loc='upper right',ncol=min(len(names),4))
        for a,key,label in zip(ax[1:],['R_A','R_sigma','rho_F'],[r'$R_A$',r'$R_\sigma$',r'$\rho(F)$']):
            finite_values=[]
            for method in methods if key!='rho_F' else methods[:1]:
                rs=[r for r in rows if r['method']==method]
                values=np.asarray([r.get(key,np.nan) for r in rs],dtype=float)
                finite_values.extend(values[np.isfinite(values)])
                color=colors[method] if key!='rho_F' else '0.2'
                marker='o' if key=='rho_F' or method=='M2' else 'x'
                a.plot([r['endpoint_s'] for r in rs],values,
                       color=color,ls=styles[method],marker=marker,ms=3,
                       markerfacecolor='none',markeredgewidth=.75,label=method)
            a.set_ylabel(label)
            # Let the estimates determine the axis range, not the threshold.
            # Include the reference at one only when it falls inside that range.
            if finite_values:
                lo,hi=min(finite_values),max(finite_values)
                pad=max(.12*(hi-lo),.015*max(abs(lo),abs(hi)),1e-6)
                lower,upper=max(0.,lo-pad),hi+pad
                a.set_ylim(lower,upper)
                if lower<=1.<=upper:
                    a.axhline(1.,color='0.55',ls=':',lw=.9,zorder=0)
            else:
                a.text(.5,.5,'No finite estimates',ha='center',va='center',transform=a.transAxes)
        ax[1].legend(loc='best')
        for i,a in enumerate(ax):
            format_axis(a,nbx=6,nby=4);panel_label(a,chr(97+i),x=-.12,y=1.02)
            # Keep the raw record and diagnostic endpoints on a common time axis.
            # A small right margin prevents clipping the last marker.
            a.set_xlim(t[0],t[-1]+dt+.01*(t[-1]+dt-t[0]))
        ax[-1].set_xlabel('Time (s)')
        save_figure(fig,out,'recording_analysis')
        if directions:
            fig,ax=plt.subplots(2,1,sharex=True,figsize=(6.5,3.6))
            fig.subplots_adjust(left=.14,right=.97,top=.94,bottom=.20,hspace=.30)
            x=np.arange(len(names)); present=[m for m in methods if any(r['method']==m for r in directions)]
            width=.7/len(present)
            for a,key,label in zip(ax,['reaction_weight','input_weight'],[r'$|r_i|$',r'$|n_i|$']):
                for j,m in enumerate(present):
                    lookup={r['channel']:r[key] for r in directions if r['method']==m}
                    a.bar(x+(j-(len(present)-1)/2)*width,[lookup.get(n,np.nan) for n in names],width=width,color=colors[m],label=m)
                a.set_ylabel(label); a.set_ylim(0,1.05)
                a.tick_params(axis='both',direction='in',top=True,right=True)
            ax[0].legend(loc='best');ax[-1].set_xticks(x);ax[-1].set_xticklabels(names,rotation=30 if len(names)>4 else 0,ha='right' if len(names)>4 else 'center')
            ax[-1].set_xlabel('Channel');panel_label(ax[0],'a',x=-.10,y=1.02);panel_label(ax[1],'b',x=-.10,y=1.02)
            save_figure(fig,out,'directions_last_window')


def main(a):
    if a.no_time_column and a.time_column!='time': raise ValueError('Choose one timestamp mode')
    path=Path(a.csv).resolve(); t,X,names,dt=load_recording(path,None if a.no_time_column else a.time_column,a.channels,a.dt)
    methods=['M2','M3'] if a.method=='both' else [a.method]
    rows,directions,last=analyze_windows(t,X,names,dt,a.window,a.step,a.lam_rel,a.shrinkage,a.centering,methods)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    fields=['start_index','end_index','last_sample_time_s','endpoint_s','method','status','error','rho_F','stable_full','gram_condition','R_A','R_sigma','rho_Gamma','stable_reduced','rate_status','plane_identifiable','direction_identifiable','support','capture_fraction','M2_M3_angle_deg']
    write_csv(out/'window_diagnostics.csv',rows,fields)
    write_csv(out/'directions_last_window.csv',directions,['method','channel','endpoint_s','reaction_weight','input_weight'])
    summary=dict(nndr_version=nndr.__version__,source_file=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),channels=names,samples=X.shape[1],dt=dt,window=a.window,step=a.step,window_seconds=a.window*dt,centering=a.centering,lam_rel=a.lam_rel,shrinkage=a.shrinkage,methods={},last_window_endpoint_s=float(t[last-1]+dt),direction_rows=len(directions))
    for m in methods:
        rr=[r for r in rows if r['method']==m]; entry=dict(windows=len(rr),successful=sum(r['status']=='ok' for r in rr),stable_full=sum(r.get('stable_full',False) for r in rr),stable_reduced=sum(r.get('stable_reduced',False) for r in rr),failed=sum(r['status'] in ('fit_failed','extraction_failed') for r in rr),unidentifiable=sum(r['status']=='plane_unidentifiable' for r in rr))
        for key in ['R_A','R_sigma','support','M2_M3_angle_deg']:
            v=np.array([r.get(key,np.nan) for r in rr],float);v=v[np.isfinite(v)]
            entry[key+'_finite']=len(v);entry[key+'_median']=float(np.median(v)) if len(v) else None
        summary['methods'][m]=entry
    (out/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Do not leave a stale directions figure after a failed final window.
    if not directions:
        for ext in ('png','svg','pdf'):
            p=out/('directions_last_window.'+ext)
            if p.exists(): p.unlink()
    plot(out,t,X,names,rows,directions,last,methods,dt)
    print(json.dumps(clean(summary),indent=2,allow_nan=False)); print('Saved:',out)
    if not any(r['status']=='ok' for r in rows): raise SystemExit('No successful extraction; inspect window_diagnostics.csv')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',default=str(HERE/'data/demo_recording.csv'))
    p.add_argument('--time-column',default='time');p.add_argument('--no-time-column',action='store_true')
    p.add_argument('--channels',nargs='+');p.add_argument('--dt',type=float)
    p.add_argument('--window',type=int,default=600,help='samples');p.add_argument('--step',type=int,default=150,help='samples')
    p.add_argument('--method',choices=['M2','M3','both'],default='both')
    p.add_argument('--lam-rel',type=float,default=.003);p.add_argument('--shrinkage',type=float,default=0.)
    p.add_argument('--centering',choices=['window','none'],default='window')
    p.add_argument('--out',default='results/examples/05');main(p.parse_args())
