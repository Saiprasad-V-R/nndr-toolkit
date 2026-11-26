"""Self-contained NNDR tutorial: no external data needed."""
import argparse
from pathlib import Path
import csv

import numpy as np
import nndr


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dt = .1
    L = nndr.complex_generator(1., 2., 1.5)
    F, Q_true = nndr.embedded_map(L, dimension=12, dt=dt, seed=1)
    X = nndr.simulate_trajectory(F, samples=5000, seed=2)
    truth = nndr.threshold_diagnostics(nndr.project_operator(F, Q_true), dt=dt)
    Fhat, info = nndr.fit_relative_operator(X, lam_rel=.003, shrinkage=.04,
                                          return_info=True)
    print('Reference ratios:', truth['R_A'], truth['R_sigma'])
    print('Estimation:', info)
    results = {}
    for method in ('M2', 'M3'):
        r = nndr.analyze_operator(Fhat, method=method, dt=dt)
        results[method] = r
        print(r.summary(), 'R_sigma:', r.R_sigma)
        print('  planted plane angle (deg):', nndr.principal_angle(r.Q, Q_true))
        print('  closure:', nndr.closure_error(Fhat, r.Q))
        print('  mean-field alignment:', nndr.plane_correlation(X, r.Q))
    print('M2/M3 plane angle:', nndr.principal_angle(results['M2'].Q, results['M3'].Q))
    r = results['M3']
    a = nndr.model_adequacy(X, Fhat, r.Q, dt=dt, lam_rel=.003, shrinkage=.04)
    print('Prediction skill:', a['prediction'])
    print('Lag errors:', a['lag_errors'])
    P, Y = nndr.adjacent_pairs(X)
    E = r.Q.T @ (Y-Fhat@P)
    print('Residual whiteness (unadjusted):', nndr.portmanteau_test(E, max_lag=5))
    if r.diagnostics['stable_full']:
        null = nndr.variance_matched_surrogate(Fhat, X, seed=10)
        print('Normal null rho, sigma:', nndr.spectral_radius(null['F']), nndr.sigma_max(null['F']))
        print('Target and stationary total variance:', null['target_variance'],
              np.trace(null['stationary_covariance']))
    print('Optimal gains at one and five steps:', nndr.optimal_gain(Fhat, 1),
          nndr.optimal_gain(Fhat, 5))
    rows = nndr.rolling_analyze(X, window=1000, step=250, method='M3',
                               dt=dt, lam_rel=.003, shrinkage=.04)
    assert len(rows) == 17
    with (out/'rolling_diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array([row['t_end'] for row in rows])*dt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for key in ('R_A', 'R_sigma'):
        ax.plot(t, [row[key] for row in rows], marker='o', label=key)
    ax.axhline(1, color='0.4', linestyle='--', linewidth=1)
    ax.set(xlabel='Exclusive window end time', ylabel='Amplification ratio')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'rolling_diagnostics.png', dpi=180)
    plt.close(fig)
    print('Saved:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/tutorial')
    main(parser.parse_args().out)
