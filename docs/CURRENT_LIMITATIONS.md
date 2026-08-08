# Current Limitations

- Final paper evaluation has not started. Reserved seeds `200-209` remain unopened.
- The action-conditioned cover observation model is not frozen.
- Task-cost weights and the grasp commitment gate are not frozen.
- Current camera actions use fixed semantic poses rather than continuous viewpoint optimization.
- Qwen raw logits are not probabilities. Only explicitly calibrated components may update the belief state.
- Relation geometry uses simulation-specific container dimensions. Lab dimensions are required for transfer.
- RG6 fingertip, lid, friction, force, and drive parameters are provisional.
- Simulation uses UR10e, while the lab has reported a UR10. The exact arm revision must be checked.
- The wrist-camera model and hand-eye transform have not been matched to the lab.
- Real-robot validation has not started.
- The current positive covered-target calibration set is small.
- A successful development episode does not establish robustness or paper-level statistical significance.
- No result should be described as transfer-ready until the hardware worksheet, frame checks, and supervised robot tests pass.
