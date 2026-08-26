# Experiments

Each weekly directory will contain frozen configurations and public
entrypoints. Large run outputs and checkpoints remain outside Git.

Week 3's `mlp_training_v1.toml` freezes the scalable learned-context MLP,
one hundred million-prediction SGD schedule, native-validation milestones, and
checkpoint events. Production runs remain operator-gated and local.

`week_03/mlp_one_epoch_continuation_v1.toml` is a separate exploratory
continuation contract. It freezes the three immutable 100M CPU parents, fixed
0.01 learning rate, first-epoch endpoint, aligned diagnostic events, Week 2
readiness aggregate, output root, and the three-seed-only decision threshold.
It does not alter the frozen primary configuration or authorize replacement of
the primary runs.

`week_03/mlp_capacity_screen_v1.toml` is a separate CPU-only exploratory
screen. It fixes three one-axis capacity arms at 25M predictions, aligned
native-validation/checkpoint events, the Week 2 readiness population, and
control provenance. It writes only ignored local run artifacts and never
selects an arm or produces a decision report automatically.

`week_03/mlp_context20_100m_continuation_v1.toml` freezes the approved
continuation of the three winning C=20 capacity-screen checkpoints from 25M to
100M predictions. It pins the capacity contract, each parent status and
checkpoint byte sequence, the full historical event schedule, original 100M
control provenance, and the three-seed native-CE threshold. It is exploratory,
CPU-only, non-resumable, and produces no automatic selection or report.

`week_03/mlp_embedding64_100m_challenger_v1.toml` is the adversarial,
parameter-count-matched alternative to that C=20 continuation. It continues
only the three fixed C=10, E=64, H=800 capacity-screen parents from 25M to
100M predictions, without replaying their first 25M predictions. It pins the
parents, C20 reference status provenance, readiness evidence, native-token
population, event schedule, and symmetric three-seed interpretation boundaries.
The E64 model has 530,965 parameters versus the frozen C20 reference's
530,293. It is CPU-only, exploratory, non-resumable, writes ignored local
artifacts, and cannot automatically choose an arm or produce a report.
