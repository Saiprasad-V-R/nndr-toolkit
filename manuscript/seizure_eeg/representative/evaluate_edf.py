"""Reconstruct the archived Figure 12 protocol; compare against frozen CSVs.
No replacement of archived values. Requires NNDR, MNE, SciPy, pandas.
"""
from pathlib import Path
import argparse, json, re, shutil, subprocess, sys, hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly
import nndr
from nndr.thresholds import threshold_diagnostics

HERE = Path(__file__).resolve().parent
CORE16 = ('FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
          'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2')
ONSET, OFFSET, FS, NATIVE = 2996, 3036, 32, 256

def base(name):
    return re.sub(r'-\d+$','',re.sub(r'^EEG\s+','',name.strip().upper()))

def read_edf(path):
    import mne
    raw=mne.io.read_raw_edf(str(path),preload=False,verbose='ERROR')
    if raw.info['sfreq'] != NATIVE: raise ValueError('Expected 256 Hz')
    found={}
    for name in raw.ch_names: found.setdefault(base(name),name)
    if any(ch not in found for ch in CORE16): raise ValueError('Missing fixed core16 channel(s)')
    selected=[found[ch] for ch in CORE16]
    x=raw.get_data(picks=selected)
    b,a=iirnotch(60,Q=30.,fs=NATIVE)
    x=filtfilt(b,a,x,axis=1)
    x=sosfiltfilt(butter(5,[.5,15.],btype='bandpass',fs=NATIVE,output='sos'),x,axis=1)
    x=resample_poly(x,up=1,down=8,axis=1)*1e6
    if not np.isfinite(x).all(): raise ValueError('Nonfinite signal')
    return x,selected

def historical_m2(F):
    # Stage-6 reference: one deterministic RNG stream, 12 restarts, 200 iterations.
    rng=np.random.default_rng(0); best=-np.inf; Q=None
    for _ in range(12):
        v=rng.normal(size=len(F)); v/=np.linalg.norm(v)
        for _ in range(200):
            z=F@v; z-=v*(v@z); norm=np.linalg.norm(z)
            if norm<1e-12: break
            u=z/norm; z=F.T@u; z-=u*(u@z); norm=np.linalg.norm(z)
            if norm<1e-12: break
            nv=z/norm
            if min(np.linalg.norm(v-nv),np.linalg.norm(v+nv))<1e-9:
                v=nv; break
            v=nv
        z=F@v; z-=v*(v@z); val=np.linalg.norm(z)
        if val>best and val>1e-12:
            best=float(val); Q=np.linalg.qr(np.column_stack((z/val,v)))[0][:,:2]
    if Q is None: raise ValueError('M2 plane undefined')
    return Q

def entropy_support(r):
    p=np.asarray(r)**2; p=p/p.sum(); positive=p>0
    return float(np.exp(-np.sum(p[positive]*np.log(p[positive])))/len(p))

def multiplier_delta(G):
    a,b=np.linalg.eigvals(G); den=abs(a+b)
    return float(abs(a-b)/den) if den>1e-12 else np.nan

def smooth(a):
    a=pd.Series(a,dtype=float); g=(a.isna()!=a.isna().shift()).cumsum()
    # Preserve gaps and omit an isolated valid point from the median overlay.
    return a.groupby(g).transform(lambda s:s.rolling(5,center=True,min_periods=min(2,len(s))).median() if len(s)>1 else s*np.nan)

def evaluate(x, ref):
    rows=[]
    for t in ref.window_end_relative_s:
        end=round((ONSET+t)*FS); start=end-40*FS
        if start<0 or end>x.shape[1]: raise ValueError('Missing window samples')
        y=x[:,start:end]; y=y-y.mean(axis=1,keepdims=True)
        F=nndr.fit_relative_operator(y,lam_rel=.003,shrinkage=.05)
        rho=nndr.spectral_radius(F); full=rho<1-1e-10
        m3=nndr.method3(F,dt=1/FS); q3=m3['Q']
        row=dict(window_start_absolute_s=start/FS,window_end_absolute_s=end/FS,
                 window_end_relative_s=t,full_stable=full,rho_F=rho)
        for method,Q in [('M2',historical_m2(F)),('M3',q3)]:
            G=Q.T@F@Q; d=threshold_diagnostics(G,dt=1/FS)
            stable=d['rhoGamma']<1-1e-10
            # Archive retains support/splitting even in reduced-unstable windows.
            metrics=dict(RA=d['R_A'] if stable and full else np.nan,
                         Rsigma=d['R_sigma'] if stable and full else np.nan,
                         support=entropy_support(Q[:,0]),delta=multiplier_delta(G))
            row[method+'_status']='ok' if stable and full and d['rate_admissible'] else ('full_unstable' if not full else 'reduced_unstable' if not stable else d['rate_status'])
            for k,v in metrics.items():
                row[f'{k}_{method}_raw']=v; row[f'{method}_{k}_valid']=bool(np.isfinite(v))
        row['support_M3_raw_commutator_basis']=entropy_support(m3['Q_comm'][:,0])
        rows.append(row)
    w=pd.DataFrame(rows)
    for k in ('RA','Rsigma','support','delta'):
        for m in ('M2','M3'): w[f'{k}_{m}_smoothed']=smooth(w[f'{k}_{m}_raw'])
    return w

def compare(ref, trial):
    if not np.array_equal(ref.window_end_relative_s,trial.window_end_relative_s): raise ValueError('Time grid mismatch')
    out={}
    for k in ['rho_F']+[c for c in ref if c.endswith('_raw')]:
        a=ref[k].to_numpy(float); b=trial[k].to_numpy(float); both=np.isfinite(a)&np.isfinite(b)
        err=np.abs(a[both]-b[both]); mismatch=int(np.sum(np.isfinite(a)!=np.isfinite(b)))
        out[k]=dict(reference_finite=int(np.isfinite(a).sum()),trial_finite=int(np.isfinite(b).sum()),mask_mismatch=mismatch,common=int(both.sum()),max_abs_error=float(err.max()) if err.size else None,median_abs_error=float(np.median(err)) if err.size else None,parity=bool(err.size and mismatch==0 and np.allclose(a[both],b[both],rtol=1e-6,atol=1e-7)))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--edf',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve()
    if out==HERE or HERE in out.parents: raise ValueError('Output must be outside frozen representative directory')
    if a.edf.name!='chb01_03.edf': raise ValueError('Expected chb01_03.edf')
    out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    ref=pd.read_csv(HERE/'data/chb01_03_window_diagnostics.csv'); e=pd.read_csv(HERE/'data/chb01_03_eeg_mean_field.csv')
    x,selected=read_edf(a.edf); trial=evaluate(x,ref)
    ix=np.rint(e.time_absolute_s.to_numpy()*FS).astype(int)
    eeg=e.copy();eeg['eeg_mean_field']=x[:,ix].mean(axis=0)
    trial.to_csv(out/'data/chb01_03_window_diagnostics.csv',index=False);eeg.to_csv(out/'data/chb01_03_eeg_mean_field.csv',index=False)
    meta=pd.read_csv(HERE/'data/chb01_03_event_metadata.csv');meta['source_artifact']='EDF reconstruction candidate';meta['annotation_source']='CHB-MIT chb01-summary.txt';meta['max_ratio_parity_abs_error']=np.nan
    meta.to_csv(out/'data/chb01_03_event_metadata.csv',index=False)
    checks=compare(ref,trial); wave=float(np.max(np.abs(e.eeg_mean_field-eeg.eeg_mean_field)))
    report=dict(status='parity_pass' if all(v['parity'] for v in checks.values()) and wave<=1e-4 else 'parity_not_established',channels=selected,edf_sha256=hashlib.sha256(a.edf.read_bytes()).hexdigest(),waveform_max_abs_error_uV=wave,metrics=checks,notes='Historical Stage-6 preprocessing and M2; current directed M3; exact original single-event analysis code not supplied. Do not substitute for archived figure unless parity verified.')
    (out/'parity_report.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copytree(HERE/'scripts',out/'scripts',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('plot_figure12_RA_main.py','plot_figure12_chb01_03_Rsigma_supp.py'): subprocess.run([sys.executable,str(out/'scripts'/name)],check=True)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
