from pathlib import Path
import pickle
from scipy.io import wavfile

DATA_DIR = Path("blizzard_wav")
CHUNK_SIZE = 200

files = sorted(DATA_DIR.glob("*.wav"))

chunk = []
chunk_idx = 0

for n, wav_path in enumerate(files, start=1):
    sr, data = wavfile.read(wav_path)
    chunk.append(data)

    if len(chunk) >= CHUNK_SIZE:
        output_file = f"data_{chunk_idx:04d}.pkl"

        print(f"Dumping chunk {chunk_idx} at file {n} of {len(files)}")

        with open(output_file, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

        chunk_idx += 1
        chunk.clear()

# Dump remaining data
if chunk:
    output_file = f"data_{chunk_idx:04d}.pkl"

    print(f"Dumping final chunk {chunk_idx}")

    with open(output_file, "wb") as f:
        pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
