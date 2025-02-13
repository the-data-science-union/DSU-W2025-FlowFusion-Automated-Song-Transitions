#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <playlist_id>"
    exit 1
fi

playlist_id=$1

echo "Running webscraper.py with playlist_id: $playlist_id"
python data/data-processing/webscraper.py $playlist_id

for file in data/raw-wavs/*.wav; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        python data/data-processing/encodings.py "$file"
    else
        echo "No .wav files found in data/raw-wavs"
        exit 1
    fi
done

echo "Script execution completed."