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
