**INCONCLUSIVE**

| screen/regime | entities | mean incremental | (a) above floor | (b) both p<0.05 | (c) majority | (d) blind drop | (e) survives projection | (f) robust contrast | (g) low similarity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all/E1-A | 242 | +0.0355 | yes | no | yes | no | no | no | no |
| all/E1-N | 228 | +0.0426 | yes | no | yes | yes | yes | yes | no |
| hatu/E1-A | 233 | -0.0115 | no | no | yes | yes | yes | n/a | no |
| hatu/E1-N | 224 | +0.0205 | yes | no | yes | yes | yes | n/a | no |

Attempted 150 conditions, 0 failed.
Smallest planted interaction the pipeline resolves on both E1 regimes: scale 1.

### The same rule with one statistic corrected (post-hoc)

Single change: the per-entity statistic's denominator is the fold's baseline MSE rather than the entity's own, which is not bounded away from zero.

**ANALOGUE-ONLY CHEMICAL TRANSFER**

| screen/regime | entities | mean incremental | (a) above floor | (b) both p<0.05 | (c) majority | (d) blind drop | (e) survives projection | (f) robust contrast | (g) low similarity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all/E1-A | 242 | +0.0419 | yes | yes | yes | no | no | no | no |
| all/E1-N | 228 | +0.0303 | yes | yes | yes | yes | yes | yes | no |
| hatu/E1-A | 233 | +0.0563 | yes | yes | yes | yes | yes | n/a | yes |
| hatu/E1-N | 224 | +0.0298 | yes | yes | yes | yes | yes | n/a | no |

Attempted 150 conditions, 0 failed.
Smallest planted interaction the pipeline resolves on both E1 regimes: scale 1.
