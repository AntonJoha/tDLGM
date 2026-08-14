import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import windows


def p2r(magnitude, phase):
    return magnitude * np.exp(1j * phase)


def r2p(x):
    return np.abs(x), np.angle(x)


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def segment_axis(x, frame_size, overlap=0):
    step = frame_size - overlap
    n_frames = (len(x) - overlap) // step

    frames = np.stack(
        [x[i * step:i * step + frame_size]
         for i in range(n_frames)],
        axis=0
    )
    return frames


class BlizzardDataset(Dataset):
    def __init__(
        self,
        data,
        x_mean=None,
        x_std=None,
        seq_len=32000,
        frame_size=200,
        use_window=False,
        use_spec=False
    ):
        self.data = data
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.use_window = use_window
        self.use_spec = use_spec

        self.x_mean = x_mean
        self.x_std = x_std

        if use_spec:
            if not is_power2(frame_size):
                raise ValueError(
                    "frame_size must be a power of 2 when use_spec=True"
                )

            self.overlap = frame_size // 2
            self.window = windows.hann(frame_size).astype(np.float32)
        elif use_window:
            self.overlap = frame_size // 2
            self.window = windows.hann(frame_size).astype(np.float32)
        else:
            self.overlap = 0

    def __len__(self):
        return len(self.data)

    def apply_window(self, batch):
        return self.window * batch

    def apply_fft(self, batch):
        batch = torch.tensor(batch, dtype=torch.float32)

        return torch.fft.rfft(batch, dim=-1)

    def log_magnitude(self, fft_batch):
        mag = torch.abs(fft_batch)
        phase = torch.angle(fft_batch)

        log_mag = torch.log10(mag + 1.0)

        return torch.polar(log_mag, phase)

    def __getitem__(self, idx):
        x = np.asarray
