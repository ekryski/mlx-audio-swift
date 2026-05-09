#!/usr/bin/env python3
"""
Extract audio samples from the pipecat-ai/stt-benchmark-data dataset.
Converts Parquet-embedded audio to WAV files and generates a benchmark manifest.

Usage:
    python3 extract_pipecat_data.py [--output-dir DIR] [--max-samples N]

Requirements:
    pip install datasets soundfile
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract pipecat STT benchmark data")
    parser.add_argument("--output-dir", default="pipecat",
                       help="Output directory for WAV files (default: pipecat/)")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum number of samples to extract (default: 100)")
    parser.add_argument("--manifest-file", default="stt-benchmark-pipecat.json",
                       help="Output manifest filename (default: stt-benchmark-pipecat.json)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("ERROR: Required packages not installed. Run:")
        print("  pip install datasets soundfile")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading pipecat-ai/stt-benchmark-data dataset...")
    dataset = load_dataset("pipecat-ai/stt-benchmark-data", "default", split="train")

    samples = []
    count = min(args.max_samples, len(dataset))
    print(f"Extracting {count} samples...")

    for i in range(count):
        item = dataset[i]
        sample_id = item["sample_id"]
        transcription = item["transcription"]
        duration = item["duration_seconds"]
        audio = item["audio"]

        # Extract audio data
        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]

        # Save as WAV
        wav_filename = f"pipecat_{i:04d}.wav"
        wav_path = os.path.join(args.output_dir, wav_filename)

        sf.write(wav_path, audio_array, sample_rate, subtype="PCM_16")

        samples.append({
            "id": f"pipecat_{i:04d}",
            "audioFile": os.path.join(args.output_dir, wav_filename),
            "referenceText": transcription,
            "domain": "clean",  # Pipecat data doesn't have domain labels
            "condition": "clean",
            "durationSeconds": round(duration, 2),
            "source": "pipecat"
        })

        if (i + 1) % 10 == 0:
            print(f"  Extracted {i + 1}/{count} samples")

    # Write manifest
    manifest = {
        "version": "1.0",
        "samples": samples
    }

    manifest_path = args.manifest_file
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Extracted {len(samples)} samples")
    print(f"  Audio files: {args.output_dir}/")
    print(f"  Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
