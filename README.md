# tdlgm :)

Run the time-series example with tuning and logs:

```bash
PYTHONPATH=src python -m tdlgm.main --tune --verbose
```

The example runs a small Optuna sweep before fitting the final model.

Skip tuning with:

```bash
PYTHONPATH=src python -m tdlgm.main --no-tune
```

Add `--verbose` to show training and tuning logs in either mode.
