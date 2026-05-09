#!/bin/bash
# benchmark.sh — Audio benchmark driver for mlx-audio-swift.
#
# Modeled on `mlx-swift-lm/scripts/benchmark.sh`. Builds the test target in
# release mode (so timings reflect compile optimizations), then iterates over
# the (pipeline × model × quant × workload) cartesian product, exporting
# `MLX_AUDIO_BENCH_*` env vars and re-invoking `swift test --skip-build` per
# permutation. Each invocation appends rows to the shared markdown +
# JSON-state pair under `benchmarks/`.
#
# Usage: ./scripts/benchmark.sh [OPTIONS]
#   --pipeline PIPELINES  comma-list: stt | tts | codec | vad | lid | sts (required)
#   --model NAMES         comma-list of registry shortnames or HF repo IDs
#   --quant QUANTS        comma-list of variant tags (bf16, 4bit, 8bit, …)
#   --kv KVS              comma-list of KV cache strategies (LLM-style models only)
#   --workload WORKLOADS  comma-list of workload identifiers
#   --batch N             batch size for STT (default 1)
#   --voice NAME          TTS voice override (e.g. af_heart)
#   --language CODE       BCP-47 language hint
#   --warmup N            warmup runs (default 1)
#   --runs N              timed runs (default 1)
#   --max-samples N       cap fixtures per run
#   --samples IDS         comma-list of fixture IDs to filter to
#   --manifest PATH       override manifest path
#   --quick               shortcut: --runs 1 --max-samples 3
#   --debug               build debug instead of release (warning: misleading timings)
#   -h, --help            show this help
#
# Examples:
#   ./scripts/benchmark.sh --pipeline stt --model parakeet-tdt-0.6b-v2 --quant bf16
#   ./scripts/benchmark.sh --pipeline tts --model kokoro-82m --voice af_heart --quick
#   ./scripts/benchmark.sh --pipeline codec --model snac-24khz,mimi
#   ./scripts/benchmark.sh --pipeline stt --model parakeet-tdt-0.6b-v2 \
#                          --workload transcription --warmup 2 --runs 3

set -e

# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────
PIPELINES=""
MODELS=""
QUANTS=""
KVS=""
WORKLOADS=""
BATCH=""
VOICE=""
LANGUAGE=""
WARMUP=""
RUNS=""
MAX_SAMPLES=""
SAMPLES=""
MANIFEST=""
QUICK=false
DEBUG=false

show_help() {
    sed -n '2,30p' "$0" | sed 's/^# //;s/^#//'
}

# ─────────────────────────────────────────────
# Parse args
# ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline)     PIPELINES="$2"; shift 2 ;;
        --model)        MODELS="$2"; shift 2 ;;
        --quant)        QUANTS="$2"; shift 2 ;;
        --kv)           KVS="$2"; shift 2 ;;
        --workload)     WORKLOADS="$2"; shift 2 ;;
        --batch)        BATCH="$2"; shift 2 ;;
        --voice)        VOICE="$2"; shift 2 ;;
        --language)     LANGUAGE="$2"; shift 2 ;;
        --warmup)       WARMUP="$2"; shift 2 ;;
        --runs)         RUNS="$2"; shift 2 ;;
        --max-samples)  MAX_SAMPLES="$2"; shift 2 ;;
        --samples)      SAMPLES="$2"; shift 2 ;;
        --manifest)     MANIFEST="$2"; shift 2 ;;
        --quick)        QUICK=true; shift ;;
        --debug)        DEBUG=true; shift ;;
        -h|--help)      show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [ -z "$PIPELINES" ]; then
    echo "Error: --pipeline is required (one of: stt, tts, codec, vad, lid, sts)" >&2
    show_help
    exit 1
fi

if [ -z "$MODELS" ]; then
    echo "Error: --model is required" >&2
    exit 1
fi

if $QUICK; then
    RUNS="${RUNS:-1}"
    MAX_SAMPLES="${MAX_SAMPLES:-3}"
fi

# ─────────────────────────────────────────────
# Locate project root + build
# ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="release"
$DEBUG && CONFIG="debug"

if $DEBUG; then
    echo "[bench] WARNING: --debug build — timings will be misleading"
fi

echo "[bench] Building (make build-tests CONFIG=$CONFIG)..."
if ! make CONFIG="$CONFIG" build-tests; then
    echo "[bench] build failed — re-run 'make build-tests' to see errors"
    exit 1
fi

# ─────────────────────────────────────────────
# Stable env vars (apply to all permutations)
# ─────────────────────────────────────────────
[ -n "$BATCH" ]       && export MLX_AUDIO_BENCH_BATCH="$BATCH"        || unset MLX_AUDIO_BENCH_BATCH
[ -n "$VOICE" ]       && export MLX_AUDIO_BENCH_VOICE="$VOICE"        || unset MLX_AUDIO_BENCH_VOICE
[ -n "$LANGUAGE" ]    && export MLX_AUDIO_BENCH_LANGUAGE="$LANGUAGE"  || unset MLX_AUDIO_BENCH_LANGUAGE
[ -n "$WARMUP" ]      && export MLX_AUDIO_BENCH_WARMUP="$WARMUP"      || unset MLX_AUDIO_BENCH_WARMUP
[ -n "$RUNS" ]        && export MLX_AUDIO_BENCH_RUNS="$RUNS"          || unset MLX_AUDIO_BENCH_RUNS
[ -n "$MAX_SAMPLES" ] && export MLX_AUDIO_BENCH_MAX_SAMPLES="$MAX_SAMPLES" || unset MLX_AUDIO_BENCH_MAX_SAMPLES
[ -n "$SAMPLES" ]     && export MLX_AUDIO_BENCH_SAMPLES="$SAMPLES"    || unset MLX_AUDIO_BENCH_SAMPLES
[ -n "$MANIFEST" ]    && export MLX_AUDIO_BENCH_MANIFEST="$MANIFEST"  || unset MLX_AUDIO_BENCH_MANIFEST

# ─────────────────────────────────────────────
# Permutation sweep
# ─────────────────────────────────────────────
IFS=',' read -ra PIPELINE_ARR <<< "$PIPELINES"
IFS=',' read -ra MODEL_ARR    <<< "$MODELS"
# Quants/KVs/workloads default to "(default)" — single-pass.
if [ -z "$QUANTS" ];    then QUANT_ARR=("");    else IFS=',' read -ra QUANT_ARR    <<< "$QUANTS"; fi
if [ -z "$KVS" ];       then KV_ARR=("");       else IFS=',' read -ra KV_ARR       <<< "$KVS"; fi
if [ -z "$WORKLOADS" ]; then WORKLOAD_ARR=(""); else IFS=',' read -ra WORKLOAD_ARR <<< "$WORKLOADS"; fi

TOTAL=$(( ${#PIPELINE_ARR[@]} * ${#MODEL_ARR[@]} * ${#QUANT_ARR[@]} * ${#KV_ARR[@]} * ${#WORKLOAD_ARR[@]} ))
INDEX=0
FAILED_RUNS=()

for pipeline in "${PIPELINE_ARR[@]}"; do
    for model in "${MODEL_ARR[@]}"; do
        for quant in "${QUANT_ARR[@]}"; do
            for kv in "${KV_ARR[@]}"; do
                for workload in "${WORKLOAD_ARR[@]}"; do
                    INDEX=$((INDEX + 1))

                    export MLX_AUDIO_BENCH_PIPELINE="$pipeline"
                    export MLX_AUDIO_BENCH_MODEL="$model"
                    [ -n "$quant" ]    && export MLX_AUDIO_BENCH_QUANT="$quant"       || unset MLX_AUDIO_BENCH_QUANT
                    [ -n "$kv" ]       && export MLX_AUDIO_BENCH_KV="$kv"             || unset MLX_AUDIO_BENCH_KV
                    [ -n "$workload" ] && export MLX_AUDIO_BENCH_WORKLOAD="$workload" || unset MLX_AUDIO_BENCH_WORKLOAD

                    desc="pipeline=$pipeline model=$model"
                    [ -n "$quant" ]    && desc="$desc quant=$quant"
                    [ -n "$kv" ]       && desc="$desc kv=$kv"
                    [ -n "$workload" ] && desc="$desc workload=$workload"
                    echo ""
                    echo "[bench] [$INDEX/$TOTAL] $desc"

                    # Stream filtered output via PTY so Swift Testing flushes
                    # mid-test. Full output captured into a tempfile for
                    # post-mortem when a run fails.
                    TMPOUT=$(mktemp)
                    if script -q /dev/null \
                            swift test --skip-build -c "$CONFIG" \
                            -Xswiftc -enable-testing \
                            --filter "MLXAudioBenchmarks" 2>&1 \
                            | tee "$TMPOUT" \
                            | grep -E --line-buffered "\[ENV\]|\[WARMUP\]|\[BENCH\]|\[MEM\]|\[RESULT\]|\[PROGRESS\]|Test.*passed|Test.*failed|[Ee]rror|[Ff]atal|threw|[Ee]xception|issue at"; then
                        :
                    fi
                    EXIT=${PIPESTATUS[0]}

                    # `script -q /dev/null` swallows the wrapped command's exit code
                    # on macOS — a Swift Testing test that throws after the test
                    # harness has reported "passed" (e.g. an MLX runtime error in a
                    # deinit) still leaves EXIT=0. Scan the captured output for
                    # known fatal markers so we don't paper over crashes.
                    POST_HOC_FAIL=0
                    # Native crashes / runtime errors that exit() cleanly past the test harness.
                    if grep -qE "MLX error|fatal error|Fatal error|terminating with uncaught exception|libc\+\+abi" "$TMPOUT"; then
                        POST_HOC_FAIL=1
                    fi
                    # Swift Testing failure markers — `Test foo failed` and
                    # `recorded an issue` both indicate a test the harness
                    # caught but `script` propagated as exit 0.
                    if grep -qE "Test .* failed|recorded an issue|with [0-9]+ issue" "$TMPOUT"; then
                        POST_HOC_FAIL=1
                    fi

                    if [ "$EXIT" -ne 0 ] || [ "$POST_HOC_FAIL" -eq 1 ]; then
                        if [ "$EXIT" -ne 0 ]; then
                            echo "[bench] run failed (exit=$EXIT): $desc"
                        else
                            echo "[bench] run failed (fatal output detected, exit=$EXIT): $desc"
                        fi
                        grep -iE "error|fatal|threw|exception|issue" "$TMPOUT" | tail -10
                        FAILED_RUNS+=("$desc")
                    fi
                    rm -f "$TMPOUT"
                done
            done
        done
    done
done

echo ""
echo "[bench] complete: $((TOTAL - ${#FAILED_RUNS[@]}))/$TOTAL runs succeeded"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "[bench] failed runs:"
    for r in "${FAILED_RUNS[@]}"; do
        echo "  - $r"
    done
    exit 1
fi
echo "[bench] reports: benchmarks/{chip}-{ram}gb-$(date +%Y-%m-%d).md"
