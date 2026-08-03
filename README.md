# tdlgm :)

Run the time-series example with tuning and logs:

```bash
PYTHONPATH=src python -m tdlgm.main --tune --verbose
```

The example runs a small Optuna sweep before fitting the final model.
Each Optuna trial reports validation loss during training, so poor runs can be pruned early.
After training, checkpoint files are saved with a timestamp and epoch counter
so repeated runs do not overwrite each other. A `config_<timestamp>.json` file
is written alongside the checkpoints with the runtime and model hyperparameters.

Skip tuning by omitting `--tune`:

```bash
PYTHONPATH=src python -m tdlgm.main
```

Add `--verbose` to show training and tuning logs in either mode.
Use `--checkpoint_interval` to control how often checkpoints are written.

Evaluate a saved checkpoint:

```bash
PYTHONPATH=src python -m tdlgm.eval --checkpoint_path artifacts/tdlgm/checkpoint.pt
```
