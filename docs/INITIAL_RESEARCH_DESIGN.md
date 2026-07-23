# Initial Research Design

## Status

The following method direction was approved as the project's initial research
design on 2026-07-24. Exact likelihood models, learned predictors, calibration
datasets, objective weights, and solver details remain experimental variables
and are not yet paper results.

## Belief and uncertainty

- Target hypotheses and task-relevant spatial relations are categorical
  probability distributions.
- Temperature scaling is the first calibration method. A probability may be
  marked calibrated only after fitting the temperature on a held-out dataset
  containing model logits and labels.
- Categorical entropy in nats is the only scalar uncertainty summary.
- Beliefs are updated by Bayesian filtering. Both positive evidence and
  negative evidence use action-conditioned likelihoods. Failure to detect an
  object should reduce its probability only to the extent that the selected
  action was expected to reveal it.

## Planning

The planner must predict possible observations before executing an action. A
planner that reads the RGB, depth, mask, or Scene Graph captured at a future
candidate pose is an offline oracle and must not be described as MPC.

The hybrid action interface contains:

- viewpoint actions;
- uncover actions;
- occluder-move actions;
- grasp actions.

Implementation is staged. The current prototype enables viewpoint and grasp.
Occluder movement is enabled in the partial-occlusion interaction stage, and
uncover is enabled in the removable-cover stage.

At every control step, the planner selects the first action of the best
finite-horizon sequence, observes the real outcome, performs a Bayesian belief
update, and replans. The initial engineering objective is

```text
expected task-failure risk
- lambda_IG * task-conditioned expected information gain
+ lambda_motion * motion cost
+ lambda_collision * collision cost
```

The current factorized engineering risk is:

```text
1 - max_h P(target = h) * P(required relation for selected target)
```

This factorization is replaceable and is not yet an approved final paper
equation.

## Current implementation boundary

`scripts/run_non_oracle_hybrid_planner.py` does not read candidate future
captures. It enumerates outcomes from a pre-action likelihood model, applies
Bayesian updates including non-detection, and evaluates horizon-two sequences.
The current likelihoods are hand-specified geometry-informed engineering
values. Therefore this is a non-oracle receding-horizon planning prototype, but
not yet the final MPC implementation.

The current initial belief is also an uncalibrated engineering stub. The
temperature-scaling code is implemented, but calibrated probabilities require
real VLM or grounding-model logits and a held-out calibration split.

Conformal prediction, CVaR, and full chance constraints are explicitly deferred
as optional future extensions.

## Reproducible command

```powershell
D:\isaac-sim\python.bat scripts\run_non_oracle_hybrid_planner.py
```

The plan is written to `outputs/non_oracle_planner/plan.json`.
