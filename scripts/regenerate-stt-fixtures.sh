#!/bin/bash
# regenerate-stt-fixtures.sh
#
# One-time utility to refresh the canonical STT fixture corpus from the
# pipecat-ai/stt-benchmark-data dataset on HuggingFace. Real recorded human
# speech beats `say`-synthesized robotic audio for STT evaluation — the
# robotic version produced false failures because models that handle real
# voices fine still tripped on the synthesis artefacts.
#
# Output: Tests/Benchmarks/Resources/stt/canonical/{*.wav,manifest.json}
#
# Usage:
#   ./scripts/regenerate-stt-fixtures.sh [--max-samples N]
#
# The companion `synthesize-stt-fixtures.sh` covers domains that pipecat
# doesn't have (URLs, code, structured numbers) using mlx-audio-swift's own
# Kokoro TTS with the af_heart voice.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/Tests/Benchmarks/Resources/stt/canonical"
MAX_SAMPLES=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --output)      OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--max-samples N] [--output DIR]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found. Install Python 3.10+ to run this script."
    exit 1
fi

echo "[fixtures] regenerating ~$MAX_SAMPLES pipecat samples into $OUTPUT_DIR"
python3 "$SCRIPT_DIR/extract_pipecat_data.py" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES"

echo "[fixtures] done. Manifest: $OUTPUT_DIR/manifest.json"
echo ""
echo "Next steps:"
echo "  - Inspect samples: ls $OUTPUT_DIR"
echo "  - For domain coverage gaps, run scripts/synthesize-stt-fixtures.sh"
