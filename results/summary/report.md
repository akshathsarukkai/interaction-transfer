All numbers below are regenerated from `results/phase1.jsonl` (563 successful runs) by `scripts/make_report.py`, which selects cells by **experimental condition** (a hash of the fully resolved run config with the seed removed), never by sweep tag. Every cell shows its own `n`.

### Which runs are in each headline cell

Selection is by condition; the tags are shown only so the provenance stays legible. A cell pools two batches iff every configured value except the seed is identical.

| pair_coverage | family        | n_seeds | seeds | contributing tags    | condition    |
| ------------- | ------------- | ------- | ----- | -------------------- | ------------ |
| 0.05          | additive      | 5       | 0–4   | main                 | 7d00aa8b0ab5 |
| 0.05          | algebra       | 5       | 0–4   | main                 | ff8bef52dedc |
| 0.05          | shared_pair   | 5       | 0–4   | main                 | 0ada562eefbc |
| 0.05          | unconstrained | 5       | 0–4   | main                 | 637fdc61543e |
| 0.10          | additive      | 17      | 0–16  | main,rep10_matched1x | c26ffacbda9c |
| 0.10          | algebra       | 17      | 0–16  | main,power10_algebra | 9d884333bc62 |
| 0.10          | shared_pair   | 17      | 0–16  | main,rep10_matched1x | 898d0ae64401 |
| 0.10          | unconstrained | 17      | 0–16  | main,rep10_matched1x | 03d6633dddff |
| 0.20          | additive      | 17      | 0–16  | main,rep020          | 83489948ed3f |
| 0.20          | algebra       | 17      | 0–16  | main,rep020          | 1be44dc9c7aa |
| 0.20          | shared_pair   | 17      | 0–16  | main,rep020          | de343e638d3e |
| 0.20          | unconstrained | 17      | 0–16  | main,rep020          | 726abda0d341 |
| 0.40          | additive      | 17      | 0–16  | cov040               | f23457da206b |
| 0.40          | algebra       | 17      | 0–16  | cov040               | 941040808436 |
| 0.40          | shared_pair   | 17      | 0–16  | cov040               | 1ac40ceb49f7 |
| 0.40          | unconstrained | 17      | 0–16  | cov040               | 1bbe2e44625d |

### Held-out-pair prediction error (headline cells)

Lower is better.

| pair_coverage | additive             | unconstrained        | shared_pair          | algebra              |
| ------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| 0.05          | 0.907 ± 0.120 (n=5)  | 0.928 ± 0.097 (n=5)  | 0.925 ± 0.166 (n=5)  | 0.921 ± 0.104 (n=5)  |
| 0.10          | 0.843 ± 0.149 (n=17) | 0.977 ± 0.169 (n=17) | 1.035 ± 0.211 (n=17) | 0.841 ± 0.168 (n=17) |
| 0.20          | 0.779 ± 0.145 (n=17) | 0.299 ± 0.112 (n=17) | 0.462 ± 0.197 (n=17) | 0.354 ± 0.126 (n=17) |
| 0.40          | 0.750 ± 0.148 (n=17) | 0.134 ± 0.033 (n=17) | 0.136 ± 0.036 (n=17) | 0.146 ± 0.052 (n=17) |

Paired `algebra − unconstrained` (matched seeds):

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.0500        | -0.0071   | 0.0616   | 5       | 0.8091  | nan        | 2                       | -0.0836 | 0.0693  |
| 0.1000        | -0.1368   | 0.1342   | 17      | 0.0007  | 0.0007     | 14                      | -0.2058 | -0.0678 |
| 0.2000        | 0.0551    | 0.0784   | 17      | 0.0106  | 0.0110     | 3                       | 0.0147  | 0.0954  |
| 0.4000        | 0.0118    | 0.0295   | 17      | 0.1179  | 0.0046     | 3                       | -0.0033 | 0.0270  |

Paired `algebra − additive` (matched seeds) — the no-interaction null:

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.0500        | 0.0138    | 0.0845   | 5       | 0.7334  | nan        | 2                       | -0.0912 | 0.1188  |
| 0.1000        | -0.0026   | 0.0905   | 17      | 0.9062  | 0.9632     | 9                       | -0.0492 | 0.0439  |
| 0.2000        | -0.4246   | 0.1240   | 17      | 0.0000  | 0.0000     | 17                      | -0.4883 | -0.3608 |
| 0.4000        | -0.6039   | 0.1244   | 17      | 0.0000  | 0.0000     | 17                      | -0.6679 | -0.5400 |

### Skill against the trained additive null (headline cells)

Higher is better.

| pair_coverage | additive             | unconstrained         | shared_pair           | algebra              |
| ------------- | -------------------- | --------------------- | --------------------- | -------------------- |
| 0.05          | 0.000 ± 0.000 (n=5)  | -0.029 ± 0.083 (n=5)  | -0.018 ± 0.108 (n=5)  | -0.021 ± 0.096 (n=5) |
| 0.10          | 0.000 ± 0.000 (n=17) | -0.170 ± 0.178 (n=17) | -0.228 ± 0.135 (n=17) | 0.001 ± 0.112 (n=17) |
| 0.20          | 0.000 ± 0.000 (n=17) | 0.617 ± 0.107 (n=17)  | 0.404 ± 0.222 (n=17)  | 0.545 ± 0.129 (n=17) |
| 0.40          | 0.000 ± 0.000 (n=17) | 0.820 ± 0.035 (n=17)  | 0.817 ± 0.043 (n=17)  | 0.805 ± 0.051 (n=17) |

Paired `algebra − unconstrained` (matched seeds):

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.0500        | 0.0077    | 0.0672   | 5       | 0.8105  | nan        | 2                       | -0.0757 | 0.0911  |
| 0.1000        | 0.1714    | 0.1856   | 17      | 0.0015  | 0.0011     | 14                      | 0.0760  | 0.2668  |
| 0.2000        | -0.0725   | 0.0928   | 17      | 0.0053  | 0.0079     | 3                       | -0.1203 | -0.0248 |
| 0.4000        | -0.0142   | 0.0319   | 17      | 0.0846  | 0.0067     | 3                       | -0.0307 | 0.0022  |

### Recovery of the symmetric interaction S (headline cells)

Higher is better.

| pair_coverage | additive | unconstrained        | shared_pair          | algebra              |
| ------------- | -------- | -------------------- | -------------------- | -------------------- |
| 0.05          | —        | 0.136 ± 0.099 (n=5)  | 0.127 ± 0.096 (n=5)  | 0.162 ± 0.149 (n=5)  |
| 0.10          | —        | 0.213 ± 0.114 (n=17) | 0.180 ± 0.080 (n=17) | 0.309 ± 0.110 (n=17) |
| 0.20          | —        | 0.739 ± 0.073 (n=17) | 0.617 ± 0.153 (n=17) | 0.706 ± 0.132 (n=17) |
| 0.40          | —        | 0.872 ± 0.027 (n=17) | 0.898 ± 0.034 (n=17) | 0.908 ± 0.031 (n=17) |

Paired `algebra − unconstrained` (matched seeds):

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.0500        | 0.0257    | 0.0955   | 5       | 0.5794  | nan        | 3                       | -0.0929 | 0.1444  |
| 0.1000        | 0.0962    | 0.0799   | 17      | 0.0001  | 0.0002     | 15                      | 0.0551  | 0.1372  |
| 0.2000        | -0.0329   | 0.0840   | 17      | 0.1256  | 0.4038     | 9                       | -0.0762 | 0.0103  |
| 0.4000        | 0.0357    | 0.0186   | 17      | 0.0000  | 0.0000     | 17                      | 0.0262  | 0.0453  |

### Recovery of the antisymmetric interaction A (headline cells)

Higher is better.

| pair_coverage | additive | unconstrained        | shared_pair          | algebra              |
| ------------- | -------- | -------------------- | -------------------- | -------------------- |
| 0.05          | —        | 0.084 ± 0.098 (n=5)  | 0.116 ± 0.045 (n=5)  | 0.114 ± 0.076 (n=5)  |
| 0.10          | —        | 0.175 ± 0.086 (n=17) | 0.172 ± 0.073 (n=17) | 0.327 ± 0.106 (n=17) |
| 0.20          | —        | 0.775 ± 0.088 (n=17) | 0.613 ± 0.189 (n=17) | 0.747 ± 0.078 (n=17) |
| 0.40          | —        | 0.909 ± 0.028 (n=17) | 0.909 ± 0.027 (n=17) | 0.892 ± 0.039 (n=17) |

Paired `algebra − unconstrained` (matched seeds):

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.0500        | 0.0299    | 0.1135   | 5       | 0.5874  | nan        | 3                       | -0.1111 | 0.1709  |
| 0.1000        | 0.1517    | 0.1172   | 17      | 0.0001  | 0.0001     | 15                      | 0.0915  | 0.2120  |
| 0.2000        | -0.0274   | 0.0741   | 17      | 0.1461  | 0.0395     | 3                       | -0.0655 | 0.0106  |
| 0.4000        | -0.0165   | 0.0347   | 17      | 0.0676  | 0.0017     | 3                       | -0.0343 | 0.0013  |

### Capacity control: every family at ~2× its headline pair parameters

`algebra`@78 (23 096 pair params) against `unconstrained`@120 (23 288). If the advantage were capacity it would not survive here.

| pair_coverage | unconstrained        | algebra              |
| ------------- | -------------------- | -------------------- |
| 0.10          | 0.964 ± 0.254 (n=17) | 0.858 ± 0.185 (n=17) |

Paired `algebra − unconstrained`:

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.1000        | -0.1058   | 0.1810   | 17      | 0.0284  | 0.0110     | 13                      | -0.1989 | -0.0127 |

S recovery:

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.1000        | 0.0676    | 0.1303   | 17      | 0.0483  | 0.0067     | 14                      | 0.0006  | 0.1346  |

A recovery:

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.1000        | 0.0917    | 0.1357   | 17      | 0.0132  | 0.0079     | 14                      | 0.0219  | 0.1614  |

### Checkpoint-selection control (final epoch vs best validation)

`final_test_mse` scores the same runs at their last epoch instead of their best-validation checkpoint. If the headline only exists at one of the two readouts it is a selection artifact.

| pair_coverage | additive             | unconstrained        | shared_pair          | algebra              |
| ------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| 0.05          | —                    | —                    | —                    | —                    |
| 0.10          | 0.845 ± 0.149 (n=17) | 0.988 ± 0.159 (n=17) | 1.076 ± 0.256 (n=17) | 0.844 ± 0.163 (n=17) |
| 0.20          | 0.781 ± 0.146 (n=17) | 0.305 ± 0.127 (n=17) | 0.478 ± 0.207 (n=17) | 0.356 ± 0.142 (n=17) |
| 0.40          | 0.750 ± 0.148 (n=17) | 0.134 ± 0.033 (n=17) | 0.135 ± 0.036 (n=17) | 0.146 ± 0.051 (n=17) |

Paired `algebra − unconstrained` at the final epoch:

| pair_coverage | mean_diff | std_diff | n_seeds | p_value | p_wilcoxon | seeds_favouring_algebra | ci95_lo | ci95_hi |
| ------------- | --------- | -------- | ------- | ------- | ---------- | ----------------------- | ------- | ------- |
| 0.0500        | nan       | nan      | 0       | nan     | nan        | 0                       | nan     | nan     |
| 0.1000        | -0.1445   | 0.1319   | 17      | 0.0004  | 0.0004     | 14                      | -0.2123 | -0.0767 |
| 0.2000        | 0.0507    | 0.1018   | 17      | 0.0568  | 0.0202     | 3                       | -0.0017 | 0.1031  |
| 0.4000        | 0.0116    | 0.0295   | 17      | 0.1253  | 0.0174     | 5                       | -0.0036 | 0.0268  |

### Controls (mean held-out MSE over seeds)

| condition                                 | coverage | additive     | unconstrained | shared_pair  | algebra      | wellspecified |
| ----------------------------------------- | -------- | ------------ | ------------- | ------------ | ------------ | ------------- |
| control_random_topology                   | 0.10     | 1.011 (n=3)  | 1.515 (n=3)   | 1.600 (n=3)  | 1.633 (n=3)  | —             |
| control_tanh_observation                  | 0.10     | 0.116 (n=3)  | 0.126 (n=3)   | 0.131 (n=3)  | 0.117 (n=3)  | —             |
| control_unmatched_capacity                | 0.10     | —            | 0.922 (n=3)   | 0.956 (n=3)  | —            | —             |
| control_weak_interactions                 | 0.10     | 0.129 (n=3)  | 0.123 (n=3)   | 0.121 (n=3)  | 0.123 (n=3)  | —             |
| misspec_0.3 + rep_misspec_0.3             | 0.10     | 0.803 (n=3)  | 1.036 (n=13)  | 0.935 (n=3)  | 0.884 (n=13) | —             |
| misspec_0.8 + rep_misspec_0.8             | 0.10     | 0.893 (n=3)  | 1.088 (n=13)  | 1.032 (n=3)  | 0.977 (n=13) | —             |
| regime_antisymmetric + rep_regime_antisym | 0.10     | 0.316 (n=15) | 0.422 (n=15)  | 0.445 (n=15) | 0.426 (n=15) | —             |
| regime_independent                        | 0.10     | 0.003 (n=3)  | 0.003 (n=3)   | 0.003 (n=3)  | 0.003 (n=3)  | —             |
| regime_symmetric + rep_regime_sym         | 0.10     | 0.528 (n=15) | 0.556 (n=15)  | 0.560 (n=15) | 0.545 (n=15) | —             |

### Parameter counts (powered headline cells)

Capacity matching is on; `control_unmatched_capacity` and the 2× control above remove it in the two opposite directions.

| family        | n_params | n_pair_params | pair_hidden |
| ------------- | -------- | ------------- | ----------- |
| additive      | 1988     | 0             | 48          |
| unconstrained | 13396    | 11408         | 76          |
| shared_pair   | 13396    | 11408         | 76          |
| algebra       | 13324    | 11336         | 48          |

### Architectural invariants (worst case over all runs)

`sym_residual` = ‖S(i,j) − S(j,i)‖, `antisym_residual` = ‖A(i,j) + A(j,i)‖. These are tripwires: they must be ~0 by construction, and a nonzero value means the implementation stopped matching the formalism.

| family        | sym_residual | antisym_residual | pair_net_order_asymmetry |
| ------------- | ------------ | ---------------- | ------------------------ |
| additive      | 0.00e+00     | 0.00e+00         | 0.00e+00                 |
| algebra       | 0.00e+00     | 0.00e+00         | 0.00e+00                 |
| shared_pair   | 0.00e+00     | 0.00e+00         | 9.92e-01                 |
| unconstrained | 0.00e+00     | 0.00e+00         | 8.08e-01                 |
| wellspecified | 8.99e-08     | 7.82e-08         | 8.99e-08                 |
