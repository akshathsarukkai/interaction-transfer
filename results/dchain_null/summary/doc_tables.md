<!-- generated: comparison_decision -->
**The decision statistics. The null unit is one simulated screen; the minimum reportable one-sided p is 1/(n+1).**

| metric | null median | null 95% interval | null max | real A375 | real PANC1 | real percentile under null | p (one-sided) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| rank-2 held-out skill @ coverage 0.40 | -0.004 | [-0.009, -0.002] | -0.002 | +0.197 | +0.161 | 100% / 100% | 0.048 / 0.048 |
| rank-2 held-out skill @ coverage 0.70 | -0.002 | [-0.013, +0.000] | +0.000 | +0.250 | +0.237 | 100% / 100% | 0.048 / 0.048 |
| rank-2 cyclic share of D² (curl fraction × top-2) | 0.071 | [0.064, 0.078] | 0.079 | 0.157 | 0.193 | 100% / 100% | 0.048 / 0.048 |
| searched low-rank skill @ coverage 0.40 | -0.001 | [-0.012, +0.001] | +0.002 | +0.229 | +0.198 | 100% / 100% | 0.048 / 0.048 |
| searched low-rank skill @ coverage 0.70 | -0.001 | [-0.006, +0.000] | +0.001 | +0.353 | +0.366 | 100% / 100% | 0.048 / 0.048 |
| rank-2 held-out skill @ coverage 0.05 | -0.034 | [-0.137, -0.005] | -0.002 | -0.051 | -0.045 | 40% / 40% | 0.619 / 0.619 |
| rank-2 held-out skill @ coverage 0.10 | -0.017 | [-0.053, -0.005] | -0.004 | -0.032 | -0.019 | 10% / 45% | 0.905 / 0.571 |
| rank-2 held-out skill @ coverage 0.20 | -0.005 | [-0.010, -0.001] | -0.001 | -0.015 | -0.029 | 0% / 0% | 1.000 / 1.000 |

<!-- generated: comparison_descriptive -->
**Descriptive — how the two worlds compare. No p-values: these are not discriminators, and the pre-registration says why.**

| metric | null median | null 95% interval | real A375 | real PANC1 |
| --- | ---: | --- | ---: | ---: |
| cyclic fraction of D  (i.i.d. noise at n=100: 0.980) | 0.788 | [0.762, 0.809] | 0.462 | 0.602 |
| curl energy in top 2  (noise floor 0.075 at k=2) | 0.091 | [0.080, 0.100] | 0.340 | 0.321 |
| curl energy in top 4  (noise floor 0.075 at k=2) | 0.169 | [0.152, 0.181] | 0.493 | 0.476 |
| curl energy in top 16  (noise floor 0.075 at k=2) | 0.507 | [0.480, 0.530] | 0.807 | 0.797 |
| spread of D (sd, off-diagonal) | 0.023 | [0.021, 0.024] | 0.223 | 0.162 |
| combination selector on-fraction | 0.199 | [0.180, 0.219] | 0.492 | 0.464 |
| posterior noise fraction of D  (see DECISION) | 4.001 | [3.665, 4.189] | 0.205 | 0.192 |
| rank-2 cyclic energy, absolute (mean D² × curl frac × top-2) | 0.000 | [0.000, 0.000] | 0.008 | 0.005 |

<!-- generated: verdict -->
**Verdict: LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT**

Computed by `report.verdict()` from 20 usable runs of the primary block (0 failed, 0 incomplete and excluded under the preregistered rule).

| clause | value |
| --- | :--: |
| null median skill ≥ half the weaker real screen, both coverages | no |
| null median rank-2 share of D² ≥ half the real mean | no |
| a real value lies inside the null 95% interval at coverage 0.70 | no |
| null median clearly positive at coverage 0.70 | no |
| null below the positive threshold, real above the null maximum, spectral below | **yes** |

| coverage | null median | null max | null 95% | real A375 | real PANC1 | artifact threshold | p (A375 / PANC1) |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.40 | -0.004 | -0.002 | [-0.009, -0.002] | +0.197 | +0.161 | +0.081 | 0.048 / 0.048 |
| 0.70 | -0.002 | +0.000 | [-0.013, +0.000] | +0.250 | +0.237 | +0.119 | 0.048 / 0.048 |

*rank-2 share of D²:* null median 0.07059, real 0.15712 (A375) / 0.19305 (PANC1).
*rank-2 cyclic energy, absolute:* null median 0.00004, real 0.00772 (A375) / 0.00503 (PANC1).
*combination selector on-fraction:* 0.1992
*split-half r(D):* 0.9579
*posterior noise fraction of D:* 4.0011
*Control A: maximum oracle rank-2 skill:* 0.0002

<!-- generated: controls -->
| block | n | true pair effect | est. synergy RMS | curl fraction | top-2 curl energy | rank-2 skill @ 0.70 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `convergence` | 4 | zero | 0.0159 | 0.780 | 0.094 | -0.002 |
| `noise_sd0.075` | 6 | zero | 0.0054 | 0.609 | 0.097 | -0.002 |
| `noise_sd0.3` | 6 | zero | 0.0434 | 0.763 | 0.080 | -0.002 |
| `oracle_nuisance` | 20 | independent, RMS 0.0604 | 0.0603 | 0.967 | 0.098 | -0.001 |
| `oracle_strict` | 20 | zero | 0.0000 | — | — | — |
| `realism` | 20 | independent, RMS 0.0604 | 0.0577 | 0.915 | 0.102 | -0.002 |
| `unshared` | 20 | zero | 0.0623 | 0.942 | 0.082 | -0.001 |

<!-- generated: mechanism -->
| block | n | offset error ε (RMS) | gain m̃ mean | m̃ sd | template R² | subspace overlap | split-half r(D) | selector on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `convergence` | 4 | 0.0054 | 0.684 | 0.246 | 0.000 | 0.010 | 0.957 | 0.192 |
| `noise_sd0.075` | 6 | 0.0023 | 0.678 | 0.241 | 0.000 | 0.009 | 0.889 | 0.135 |
| `noise_sd0.3` | 6 | 0.0129 | 0.678 | 0.241 | 0.000 | 0.011 | 0.985 | 0.312 |
| `primary` | 20 | 0.0051 | 0.670 | 0.244 | 0.000 | 0.008 | 0.958 | 0.199 |
| `realism` | 20 | 0.0056 | 0.670 | 0.244 | 0.000 | 0.015 | 0.991 | 0.313 |
