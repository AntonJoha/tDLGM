#!/usr/bin/env bash

DIRNAME="blizzard_wav"
mkdir -p "$DIRNAME"

awk '/\.mp3\.?$/ {print $NF}' out.txt | while read -r f
do
    # Add leading slash if missing
    [[ "$f" != /* ]] && f="/$f"

    t=$(basename "$f" .mp3)

    echo "Processing: $f"

    ffmpeg -y \
        -i "$f" \
        -acodec pcm_s16le \
        -ac 1 \
        -ar 16000 \
        "$DIRNAME/${t}.wav"
done
