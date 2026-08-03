# tdlgm :)

Run the time-series example with tuning and logs:

```bash
PYTHONPATH=src python -m tdlgm.main --tune --verbose
```

The example runs a small Optuna sweep before fitting the final model.
Each Optuna trial reports validation loss during training, so poor runs can be pruned early.

Skip tuning by omitting `--tune`:

```bash
PYTHONPATH=src python -m tdlgm.main
```

Add `--verbose` to show training and tuning logs in either mode.
