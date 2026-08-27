"""Phase 3 -- entity-level out-of-distribution evaluation of the Koplev screen.

Phase 2R held out *pairs*: both drugs were still seen elsewhere in the training
graph, so the model could learn a latent vector for each and complete the matrix.
Phase 3 holds out *drugs*. A test drug contributes no Koplev measurement to
anything that is fitted -- not the model, not the validation set, not a
normalisation constant, not a hyperparameter choice. The only thing the model may
know about it is what the molecule is.
"""

from __future__ import annotations
