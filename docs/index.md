# Learn NNDR

[Repository home](../README.md) · [Manuscript resources](../manuscript/README.md)

A stable system can still amplify a perturbation before it decays. NNDR estimates the local dynamics from a multivariate recording and extracts a two-dimensional plane in which to investigate this amplification. It can also be used directly with a known linear map.

## Start with your question

| I want to… | Where to start |
|---|---|
| Understand why stability does not rule out amplification | [Understanding NNDR](understanding_nndr.md) |
| Install the package and run my first analysis | [Quick start](quickstart.md) |
| Analyze my model, simulations or measurements | [User guide](user_guide.md) |
| Work through a complete synthetic example | [Tutorial](tutorial.md) |
| Understand the real and complex thresholds | [Mathematical framework](mathematical_framework.md) |
| Check whether the fitted model is reliable | [Validation and uncertainty](validation_and_uncertainty.md) |
| Find a function, argument or output field | [API reference](api_reference.md) |
| Reproduce results associated with the paper | [Manuscript resources](../manuscript/README.md) |

## The analysis at a glance

```text
Multivariate observations                  Known discrete map
          |                                         |
          v                                         |
Fit a one-step linear predictor                      |
          |                                         |
          +------------------+----------------------+
                             v
                   Extract response plane
                             |
                             v
              R_A, R_sigma, directions, support
                             |
                             v
                Assess stability and reliability
```

The two ratios answer different scientific questions. Neither establishes causation, and neither replaces a check of the full fitted operator. The [conceptual guide](understanding_nndr.md) introduces both before the derivation.

## Technical reference

- [Scientific geometry](scientific_api.md): ordered directions, alignment and support.
- [Estimation and adequacy](estimation_adequacy.md): regularization, gains, closure and predictive checks.
- [Statistical controls](statistical_controls.md): surrogate models, residual tests, bootstrap and logarithm branches.
- [Limitations](limitations.md): assumptions, identifiability, sampling and historical conventions.
- [Migration from the original version](../THRESHOLD_MIGRATION.md): changes in scientific definitions.

The current scientific API is 2.0.0. The published tag `2.0.0` has the earlier directory layout; the guides here refer to the current checkout. Follow documentation within the tag when reproducing that fixed snapshot.
