# Week 4 Experiment Records

Each experiment should record at least:

- `experiment_id`
- `date`
- `git_commit`
- `source_commit`
- `config`
- `dataset_version`
- `device`
- `training_time`
- `peak_memory`
- `throughput`
- `train_loss`
- `validation_loss` (if available)
- `checkpoint`
- `generation_samples`
- `notes`

Record `git_commit` when the experiment actually runs, using:

```bash
git rev-parse HEAD
```

Do not guess the repository commit in advance. `source_commit` identifies the MiniMind upstream commit used by the imported source.
