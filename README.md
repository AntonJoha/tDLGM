# tdlgm :)

Run the time-series example with:

```bash
PYTHONPATH=src python -m tdlgm.main --tune
```

The example runs a small Optuna sweep before fitting the final model.

Skip tuning with:

```bash
PYTHONPATH=src python -m tdlgm.main --no-tune
```
