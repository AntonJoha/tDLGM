# tdlgm :)

The model is a sequential VAE: a temporal recognition LSTM infers one Gaussian
latent variable per time step, while the generator evolves its recurrent state
from those latents. The generator also learns a conditional prior
`p(z_t | h_t)`. Training anneals the conditional KL term and gradually replaces
posterior samples with prior samples; evaluation and generation use only the
conditional prior.

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
