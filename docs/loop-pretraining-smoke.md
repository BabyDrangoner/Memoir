# Loop attention world-model pretraining smoke run

This is a new experiment, based on source commit `9098e42f6b1fb26bfc5809e265c7ebd2a86dced3`. It does not modify an existing training identity or resume the hybrid step-5000 checkpoint.

The backend is `drama_mamba2_temporal_loop`: three Mamba-2 layers followed by shared attention refinement, width 768, 12 attention heads, fixed loop count `R=2`. Mamba temporal states advance once per trajectory position. The refinement iterations read the same source K/V and update the current query/latent; the attention cache grows by the number of actual positions, not by the loop count. There is no additional FFN. This is a fixed-memory looped attention experiment, not a reproduction of every component of a published Loop Transformer.

The config `configs/loop_pretrain_smoke.json` retains the existing author-path world-model objective and its 5,000-update learning-rate schedule, including 100 warmup updates. The engineering test stops after 100 updates, then strictly restores and advances to 102. Neither stop is labelled completed pretraining; no final step-5000 handoff is exported.

| Setting | Value |
| --- | --- |
| Initialization | Author TSSM pretrained checkpoint; six world-model modules initialized fresh |
| Frozen | Encoders and all navigation-policy parameters |
| Trained | Exactly 81 parameter tensors in the six world-model modules |
| Architecture identity | Distinct backend plus persistent `prior._loop_iterations` buffer |
| Data pool | All 14,039 R2R instructions and 1,069,620 PREVALENT rows, with original file SHA checks |
| Sampling | Uniform shuffled union without replacement; no validation scoring |
| Global loss batch | 32 trajectories, shared contrastive negative pool |
| Microbatch | 2 initially, existing OOM handler may reduce to 1 |
| Overshooting horizon | 5, independent of loop count 2 |
| Loss | Existing visual contrastive, prior/posterior KL, overshoot KL and next-expert-node goal-distance MSE |
| Seed | 0 |
| Storage | Explicit `runtime-local` production backend, separate run/backup directories on the same Colab `/content` filesystem |

The data-pool counts are not a claim that all rows have been visited after 102 updates. This is a computation, gradient, storage and resume validation; it provides no SR/SPL or seen/unseen result.

Assuming the experiment directory below has been prepared with its `activate.sh`, source tree and verified assets:

```bash
memoir_experiment_root=/content/project/memoir-loop-pretrain-20261127
source "$memoir_experiment_root/activate.sh"
cd "$memoir_experiment_root/Memoir"

python -u scripts/colab_full_pretrain.py \
  --config configs/loop_pretrain_smoke.json \
  --world-model-backend drama_mamba2_temporal_loop \
  --asset-root "$memoir_experiment_root/datasets/R2R" \
  --run-root "$memoir_experiment_root/run" \
  --backup-root "$memoir_experiment_root/verified-state" \
  --backup-backend runtime-local --stop-after 100

python -u scripts/colab_full_pretrain.py \
  --config configs/loop_pretrain_smoke.json \
  --world-model-backend drama_mamba2_temporal_loop \
  --asset-root "$memoir_experiment_root/datasets/R2R" \
  --run-root "$memoir_experiment_root/run" \
  --backup-root "$memoir_experiment_root/verified-state" \
  --backup-backend runtime-local --require-resume --stop-after 102
```

Use the actual prepared root consistently if it differs from the example. Do not change the config's `steps` to 100 or 102 between these commands: that would change the schedule and identity. Strict restore rejects architecture, data, source, runtime, optimizer/scheduler or frozen-base mismatches instead of silently starting from zero.

Required evidence from the GPU run:

1. `PRETRAIN_ARCHITECTURE` records loop count 2 and fixed-source memory semantics.
2. `PRETRAIN_DATA` confirms the two complete source row counts.
3. `PRETRAIN_START` initially shows step 0 / fresh, and after restart shows step 100 / `require_resume=true` with the same identity SHA.
4. Every `PRETRAIN_PROGRESS` has finite loss and gradient norm, with the actual microbatch and peak allocated GPU memory recorded.
5. `PRETRAIN_GRADIENT_AUDIT` records finite gradients for the full world-model partition, the nonzero/zero gradient tensor names, and absent gradients on frozen parameters. Reports are written on the first fresh and first resumed updates; the checks run every update.
6. `PRETRAIN_FROZEN_AUDIT` at steps 0, 100 and 102 confirms all frozen tensor hashes, including the architecture buffer, are unchanged and binds the committed checkpoint ID.
7. Primary and backup snapshots contain the full trainable state, optimizer, scheduler, sampler and RNG, alongside the SHA-bound frozen base and identity. Mac copies need independent readback SHA verification.

The two `/content` copies share one ephemeral VM filesystem. They preserve atomic-save and readback checks but do not survive runtime replacement. Preserve the frozen base, latest full checkpoint, identity/config, logs and audit JSONs on the Mac before releasing the runtime. No validation access is needed for this smoke run.

CPU checks use real production-width loop dynamics with small observation/prediction heads. They cover the author world-model loss and overshooting, finite/nonzero gradients, frozen parameter integrity, and exact equality of the next optimizer update after restoring model/optimizer/sampler/scheduler/RNG. Core tests separately cover R=1 baseline parity, whole-sequence/chunked cache and gradient parity, causal masking and immutable imagination branches.
