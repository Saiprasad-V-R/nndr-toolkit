# Mathematical framework

[Home](../README.md) · [API reference](api_reference.md)

## 1. Estimation and reduction

Let a regularly sampled, real record be $X\in\mathbb R^{N\times T}$ and let $P=X_{:,0:T-1}$, $Y=X_{:,1:T}$. The local map is

$$\nu=\nu_{\mathrm{rel}}\mathrm{tr}(PP^\top)/N,\qquad \hat F=YP^\top(PP^\top+\nu I)^{-1}.$$

Trace-preserving shrinkage gives $F=(1-\alpha)\hat F+\alpha\mathrm{tr}(\hat F)I/N$, $0\le\alpha<1$. Uniform rescaling preserves the relative ridge estimate, unlike independent scaling of channels. No stability rescaling is applied. The estimator models one-step prediction, not necessarily the physical Jacobian.

M1 uses the eigenbasis SVD baseline. M2 seeks orthogonal unit directions maximizing $r^\top F n$ by alternating optimization. M3 uses extreme eigenvectors of the symmetric commutator $B=FF^\top-F^\top F$, then maximizes directed coupling within their span. In the ordered basis $Q=[r,n]$, $Q^\top Q=I_2$ and $\Gamma=Q^\top FQ$. M3's internal two-dimensional optimization is exact, but the plane itself is not necessarily optimal for any specified time horizon. Plane and direction identifiability are different.

## 2. Spectral geometry

For distinct multipliers $\lambda_i$ of $\Gamma$, Henrici departure and non-normality are

$$d_F(\Gamma)=\sqrt{\|\Gamma\|_F^2-|\lambda_1|^2-|\lambda_2|^2},\quad K=d_F(\Gamma)/|\lambda_1-\lambda_2|.$$

For normalized eigenvectors $p_i$, $c=|p_1^*p_2|$, $\kappa_{2D}=\sqrt{(1+c)/(1-c)}$, and $K=(\kappa_{2D}-\kappa_{2D}^{-1})/2$. A defective repeated pair has infinite $K$; a scalar repeated pair is normal. The implementation avoids eigenvector-overlap clipping. General-dimensional departure by norm subtraction can lose relative precision near normality.

## 3. Corrected thresholds

For a stable real reduced map (spectral radius below one), use the real principal logarithm $L=\log(\Gamma)/\Delta t$ when admissible. Zero and negative real multipliers are inadmissible for this logarithm. A sampled oscillation may be aliased. Distinguish $\lambda_i$ (multipliers) from $\mu_i$ (rates).

Real rates $\mu_i=-\gamma_i$, $\gamma_i>0$:

$$\Delta_{\rm rate}=\frac{|\gamma_1-\gamma_2|}{\gamma_1+\gamma_2},\quad s=\sqrt{1-\Delta_{\rm rate}^2},\quad K_{c,A}=\sqrt{\frac{s}{1-s}}.$$

Complex rates $\mu_{1,2}=-\gamma\pm i\omega$:

$$\beta=|\omega|/\gamma,\qquad K_{c,A}=\frac{1}{\sqrt{\sqrt{1+\beta^2}-1}}.$$

Away from degeneracy $R_A=K/K_{c,A}$. At positive repeated multipliers the code uses the finite divided-difference logarithm limit; for repeated decay rate $\gamma$, $R_A=d_F(L)/(2\sqrt2\gamma)$. This avoids an indeterminate infinity divided by infinity. For stable admissible maps, $R_A>1$ means the isotropic impulse variance-response kernel $A(t)=\|e^{Lt}\|_F^2$ has an increasing interval, not necessarily a peak exceeding $A(0)$. Equality is a boundary, not strict growth.

The exact discrete one-step ratio is

$$R_\sigma=\frac{d_F(\Gamma)}{\sqrt{(1-|\lambda_1|^2)(1-|\lambda_2|^2)}}.$$

For Schur-stable $\Gamma$, $R_\sigma>1$ if and only if $\sigma_{\max}(\Gamma)>1$. The ratio is also valid when the principal real logarithm is inadmissible. Neither ratio is reported for unstable reduced maps. Full-map stability is checked independently. Changing the sampling interval through a new sampled map can change the diagnostics; relabeling time units for a fixed map does not change the two ratios.

## 4. Directional and observable diagnostics

$$S(r)=\frac{\|r\|_1}{\sqrt N\|r\|_2},\qquad m(t)=N^{-1}\sum_i x_i(t),\qquad y_r=r^\top x,\quad y_n=n^\top x.$$

The maximum absolute plane-observable correlation uses centered plane coordinates $z=Q^\top x$ and least-squares projection, $a_\star\propto C_{zz}^{\dagger}c_{zm}$. It measures alignment, not held-out predictive skill. A constant observable has undefined correlation. Largest principal angle between orthonormal planes is $\arccos\sigma_{\min}(Q^\top Q_\star)$; API units are degrees. Capture fraction is $\sigma_{\max}(\Gamma)/\sigma_{\max}(F)$ and relative closure error is $\|FQ-Q\Gamma\|_F/\|FQ\|_F$.

The archived seizure figure uses distinct quantities: entropy support $\exp(-\sum_i p_i\log p_i)/N$ with $p_i=r_i^2/\|r\|_2^2$; and multiplier splitting $|\lambda_1-\lambda_2|/|\lambda_1+\lambda_2|$. These are not aliases for the default support or rate splitting. See [reproducibility](../manuscript/seizure_eeg/representative/README.md).

## 5. Dynamical response and checks

At integer lag $k$, $G(k)=\|F^k\|_2$ is maximum norm gain (not squared), $A(k)=\|F^k\|_F^2$ is the impulse covariance trace. Use $e^{Lt}$ for continuous time. Numerical abscissa is $\lambda_{\max}((L+L^\top)/2)$. The discrete Kreiss constant is $\sup_{|z|>1}(|z|-1)\|(zI-F)^{-1}\|_2$; the implementation evaluates a grid lower bound. See [adequacy](estimation_adequacy.md) and [statistical controls](statistical_controls.md) for held-out skill, lag structure, residual tests, variance-matched normal references and bootstrap uncertainty. These are building blocks, not a calibrated screening protocol.
