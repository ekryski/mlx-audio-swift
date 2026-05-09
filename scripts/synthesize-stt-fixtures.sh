#!/bin/bash
# synthesize-stt-fixtures.sh
#
# Re-synthesize the STT fixture corpus using mlx-audio-swift's own Kokoro
# 82M with the af_heart voice. Replaces the previous macOS `say`-generated
# robotic audio so STT failures reflect real model weaknesses, not synthesis
# artefacts.
#
# Reads a manifest of (id, referenceText) pairs and writes a fresh WAV per
# entry plus an updated manifest tagged with `source: "kokoro-af-heart"` so
# downstream readers can filter by provenance.
#
# This dogfoods our own TTS as a benchmark generator — a clean recursive
# use of the repo's capabilities, and the synthesized fixtures double as a
# sanity check that Kokoro produces clear, intelligible speech across edge
# cases.
#
# Usage:
#   ./scripts/synthesize-stt-fixtures.sh \
#       --input  Tests/Benchmarks/Resources/stt/canonical/manifest.json \
#       --output Tests/Benchmarks/Resources/stt/canonical/ \
#       [--voice af_heart] [--engine kokoro|echo-tts] [--model HF_REPO]
#
# Defaults: voice af_heart, engine kokoro, model mlx-community/Kokoro-82M-bf16

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

INPUT_MANIFEST="$PROJECT_ROOT/Tests/Benchmarks/Resources/stt/canonical/manifest.json"
OUTPUT_DIR="$PROJECT_ROOT/Tests/Benchmarks/Resources/stt/canonical"
VOICE="af_heart"
ENGINE="kokoro"
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)  INPUT_MANIFEST="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --voice)  VOICE="$2"; shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# //;s/^#//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default model per engine.
if [ -z "$MODEL" ]; then
    case "$ENGINE" in
        kokoro)   MODEL="mlx-community/Kokoro-82M-bf16" ;;
        echo-tts) MODEL="mlx-community/echo-tts-base" ;;
        *) echo "Unknown engine: $ENGINE (kokoro | echo-tts)"; exit 1 ;;
    esac
fi

if [ ! -f "$INPUT_MANIFEST" ]; then
    echo "Error: input manifest not found at $INPUT_MANIFEST" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Build the TTS executable.
echo "[synth] building mlx-audio-swift-tts (release)..."
cd "$PROJECT_ROOT"
swift build -c release --product mlx-audio-swift-tts >/dev/null
TTS_BIN="$PROJECT_ROOT/.build/release/mlx-audio-swift-tts"

# Iterate the manifest. Use Python (already a dep for extract_pipecat_data.py)
# rather than jq so this works on the box without needing jq installed.
python3 - "$INPUT_MANIFEST" "$OUTPUT_DIR" "$TTS_BIN" "$MODEL" "$VOICE" "$ENGINE" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

input_manifest, output_dir, tts_bin, model, voice, engine = sys.argv[1:7]
manifest_path = Path(input_manifest)
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

with manifest_path.open() as f:
    manifest = json.load(f)

samples = manifest.get("samples", [])
print(f"[synth] re-synthesizing {len(samples)} sample(s) with engine={engine} voice={voice}")

new_samples = []
for entry in samples:
    sid = entry["id"]
    text = entry.get("referenceText") or entry.get("text")
    if not text:
        print(f"[synth] skip {sid}: no text")
        continue

    out_wav = output_path / f"{sid}.wav"
    cmd = [
        tts_bin,
        "--text", text,
        "--voice", voice,
        "--model", model,
        "--output", str(out_wav),
    ]
    print(f"[synth] {sid}: {text[:60]}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[synth]  failed: {result.stderr.strip()}")
        continue

    new_entry = dict(entry)
    new_entry["audioFile"] = f"{sid}.wav"
    new_entry["source"] = f"{engine}-{voice}"
    new_samples.append(new_entry)

# Write back manifest with updated sources.
out_manifest = manifest_path
out_manifest.write_text(json.dumps(
    {"version": manifest.get("version", "1.0"), "samples": new_samples},
    indent=2
))
print(f"[synth] wrote {len(new_samples)} samples + manifest at {out_manifest}")
PY

echo "[synth] done"
