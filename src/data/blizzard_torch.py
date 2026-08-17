from pathlib import Path
import logging

from experiments.util import configure_logging
import pickle
import librosa

logger = logging.getLogger(__name__)

path = Path(__file__).parent
folder = path / "blizzard_wav"


samples_per_second = 16000
segment_length = 0.5 # seconds
dimensions = 200
horizon = 1


def get_all_wavs():
    wavs = list(folder.glob("*.wav"))
    return wavs


def make_segments(audio, sr):
    segments = []
    for i in range(0, len(audio) - dimensions, dimensions):
        segment = audio[i:i + dimensions]
        segments.append(segment)
    return segments


def get_seq_len():
    return int((samples_per_second * segment_length)/dimensions)

def segments_to_entries(segments):
    entries = []
    seq_len = get_seq_len()
    for i in range(0, len(segments) - seq_len - horizon):
        entry = segments[i:i + seq_len]
        target = segments[i + seq_len:i + seq_len + horizon]
        entries.append((entry, target))
    return entries

def load_data():
    checkpoints = 200
    count = 0
    files = get_all_wavs()
    entries = []
    for f in files:
        audio, sr = librosa.load(f, sr=None)
        segments = make_segments(audio, sr)
        entries.append(segments_to_entries(segments))
        count += 1

        if count % checkpoints == 0:
            logger.info(f"Processed {count} files")
            save_path = path / f"data_{count/checkpoints:04d}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
            entries.clear()



    

    audio, sr = librosa.load(path / "blizzard_wav/01_bx_potter_01.wav", sr=None)
    logger.info(audio.shape, sr)
    logger.info(max(audio), min(audio))



if __name__ == "__main__":
    configure_logging(True)
    load_data()

    print("Hello")
