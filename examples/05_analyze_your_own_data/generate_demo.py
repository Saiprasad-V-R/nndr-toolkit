#!/usr/bin/env python3
"""Generate the bundled synthetic sensor recording; no experimental data."""
import argparse
import csv
from pathlib import Path
import numpy as np
import nndr


def main(out):
    # Orthogonal two-group plane; stable background, directed coupling.
    q = np.array([[1.,0.],[2.,0.],[0.,2.],[0.,1.]])/np.sqrt(5.)
    G = np.array([[.8,.9],[0.,.5]])
    F = .25*np.eye(4)+q@(G-.25*np.eye(2))@q.T
    X = nndr.simulate_trajectory(F,samples=3000,noise_std=.1,burn_in=500,seed=12)
    out = Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n'); w.writerow(['time','sensor_a','sensor_b','sensor_c','sensor_d'])
        w.writerows([format(k*.05,'.12g'),*(format(v,'.16g') for v in X[:,k])] for k in range(X.shape[1]))
    print('Saved synthetic recording:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(Path(__file__).resolve().parent/'data/demo_recording.csv'))
    main(p.parse_args().out)
