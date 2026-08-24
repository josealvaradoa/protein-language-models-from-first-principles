# Experiments

Each weekly directory will contain frozen configurations and public
entrypoints. Large run outputs and checkpoints remain outside Git.

Week 3's `mlp_training_v1.toml` freezes the scalable learned-context MLP,
one hundred million-prediction SGD schedule, native-validation milestones, and
checkpoint events. Production runs remain operator-gated and local.
