| representation | screen | folds | E1-A | E1-N | E2 |
|---|---|---:|---:|---:|---:|
| ecfp4 | all | 5 | -0.0010 | +0.0245 | -0.0011 |
| ecfp4 | hatu | 5 | +0.0450 | +0.0539 | +0.0429 |
| random | all | 5 | -0.0318 ** | -0.0474 ** | -0.0236 ** |
| random | hatu | 5 | -0.0366 ** | -0.0050 ** | -0.0015 ** |
| shuffled_acid | all | 5 | -0.0312 ** | -0.0056 *off-role* | -0.0041 *off-role* |
| shuffled_acid | hatu | 5 | -0.0526 ** | +0.0233 *off-role* | -0.0167 *off-role* |
| shuffled_amine | all | 5 | +0.0159 *off-role* | +0.0397 ** | +0.0347 *off-role* |
| shuffled_amine | hatu | 5 | +0.0298 *off-role* | -0.0253 ** | -0.0061 *off-role* |
| shuffled_both | all | 5 | +0.0069 *off-role* | +0.0081 *off-role* | +0.0143 ** |
| shuffled_both | hatu | 5 | -0.0301 *off-role* | -0.0172 *off-role* | +0.0053 ** |

`**` marks a control that controls that regime. Shuffling a role destroys generalisation to *unseen* entities of that role and nothing else -- for a seen entity a shuffled fingerprint is still a unique consistent key -- so a shuffle is on-role only in the regime where the shuffled endpoint is the unseen one. `random` carries no chemistry in **either** role and is on-role everywhere. The cells marked *off-role* shuffle the endpoint the model has already trained on and are not controls; the registration did not distinguish them and is evaluated as written.
