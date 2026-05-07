# mlx-audio-swift Benchmarks

Generic, model-comparison benchmarks for every audio model family in this
repo: STT, TTS, codecs, VAD/diarization, LID, and speech-to-speech (STS).
Application-specific pipeline benchmarks (STT→LLM→TTS, agentic workloads)
live in the [Sam](https://github.com/ekryski/sam) repo. LLM-only benchmarks
live in [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm).

This harness is modeled directly on `mlx-swift-lm`'s benchmark target so
reports across the two repos look and feel the same — same chip slug, same
hardware-dated filename, same `[BENCH]/[RESULT]/[MEM]` log markers, same
JSON-sidecar idempotent re-render.

## Quick start

```bash
# Build once in release mode (required — debug timings are misleading).
make build-tests

# STT — Parakeet on the canonical pipecat-derived corpus
./scripts/benchmark.sh --pipeline stt --model parakeet-tdt-0.6b-v2 --quant bf16

# TTS — Kokoro at the af_heart voice
./scripts/benchmark.sh --pipeline tts --model kokoro-82m --voice af_heart

# Codec — SNAC encode/decode RTF + SI-SNR reconstruction quality
./scripts/benchmark.sh --pipeline codec --model snac-24khz

# VAD — Sortformer diarization
./scripts/benchmark.sh --pipeline vad --model sortformer-streaming-4spk

# LID — MMS multilingual language ID
./scripts/benchmark.sh --pipeline lid --model mms-lid-256

# STS — LFM2.5-Audio understanding (text-to-text)
./scripts/benchmark.sh --pipeline sts --model lfm25-audio-1.5b

# Quick smoke run (3 fixtures, 1 timed pass)
./scripts/benchmark.sh --pipeline stt --model parakeet-tdt-0.6b-v2 --quick
```

Reports land at `benchmarks/{chip}-{ram}gb-{YYYY-MM-DD}.md`. Re-running the
same command appends rows to the same file via the JSON state sidecar
`benchmarks/.{filename}.state.json`.

## Pipelines and metrics

| Pipeline | Headline metrics                                             |
|----------|--------------------------------------------------------------|
| `stt`    | WER, semantic-WER, CER, RTF, time-to-first-word, words/sec   |
| `tts`    | RTF (output-audio / processing-time), chars/sec, samples/sec, time-to-first-audio |
| `codec`  | encode RTF, decode RTF, SI-SNR (reconstruction quality)       |
| `vad`    | RTF, segment count (frame-accuracy / FPR / FNR when labeled)  |
| `lid`    | RTF, top-1 / top-3 accuracy (when labels supplied)            |
| `sts`    | RTF, processing time, generation preview                      |

All pipelines report peak GPU memory and resident RAM.

## Permutation sweeps

The shell driver iterates `pipeline × model × quant × kv × workload`,
exporting one `MLX_AUDIO_BENCH_*` env var per dimension. Each permutation
runs in its own `swift test --skip-build` invocation so a transient model
load failure or OOM in one cell doesn't block the rest:

```bash
./scripts/benchmark.sh \
    --pipeline stt \
    --model parakeet-tdt-0.6b-v2,qwen3-asr-0.6b \
    --quant bf16,4bit \
    --workload transcription
```

Stale rows are not removed — the report grows monotonically as you sweep.
To start a fresh report, delete the matching `.md` + `.state.json` pair.

## Environment variables

The shell driver sets these per permutation; you can also export them
directly when invoking `swift test` by hand:

| Variable                     | Description                                |
|------------------------------|--------------------------------------------|
| `MLX_AUDIO_BENCH_PIPELINE`   | `stt | tts | codec | vad | lid | sts`      |
| `MLX_AUDIO_BENCH_MODEL`      | Registry shortname or `owner/repo` ID      |
| `MLX_AUDIO_BENCH_QUANT`      | Variant tag (`bf16`, `4bit`, …)            |
| `MLX_AUDIO_BENCH_KV`         | KV cache strategy (LLM-style models only)  |
| `MLX_AUDIO_BENCH_WORKLOAD`   | Per-pipeline workload identifier           |
| `MLX_AUDIO_BENCH_LANGUAGE`   | BCP-47 language hint                       |
| `MLX_AUDIO_BENCH_BATCH`      | Batch size (STT)                           |
| `MLX_AUDIO_BENCH_VOICE`      | TTS voice override                         |
| `MLX_AUDIO_BENCH_WARMUP`     | Number of warmup runs (default 1)          |
| `MLX_AUDIO_BENCH_RUNS`       | Number of timed runs (default 1)           |
| `MLX_AUDIO_BENCH_MAX_SAMPLES`| Cap on fixtures per run                    |
| `MLX_AUDIO_BENCH_SAMPLES`    | Comma-list of fixture IDs                  |
| `MLX_AUDIO_BENCH_MANIFEST`   | Override manifest path                     |

## Fixtures

Fixtures live under `Tests/Benchmarks/Resources/{pipeline}/`. Each pipeline
has a `manifest.json` keyed by sample id with optional reference data
(transcripts, languages, turn boundaries). The manifest format is shared
across pipelines; runners read only the fields they need.

The canonical STT corpus is sourced from
[`pipecat-ai/stt-benchmark-data`](https://huggingface.co/datasets/pipecat-ai/stt-benchmark-data)
(real recorded speech) plus, for domains pipecat doesn't cover (URLs, code
snippets, structured numbers), Kokoro 82M re-synthesis using the
**af_heart** voice. Provenance is recorded in each manifest entry's
`source` field so reports can filter by origin.

To regenerate the corpus:

```bash
./scripts/regenerate-stt-fixtures.sh                  # pipecat
./scripts/synthesize-stt-fixtures.sh --voice af_heart # Kokoro fallback
```

## Adding a model

Add an entry to `Tests/Benchmarks/Utils/ModelRegistry.swift`:

```swift
ModelFamily(
    pipeline: .stt,
    name: "Display Name",
    shortName: "registry-shortname",
    variants: [
        ModelVariant(quantization: "bf16", repoId: "mlx-community/your-model"),
        ModelVariant(quantization: "4bit", repoId: "mlx-community/your-model-4bit"),
    ],
    notes: "free-form tag shown in the report's parameter table"
)
```

Then wire the loader in the matching pipeline runner — the existing cases
in `STTBenchmarkRunner.loadAndConfigure` / `TTSBenchmarkRunner.loadAndConfigure`
/ etc. are the patterns to copy.

For one-off runs without modifying the registry, pass an `owner/repo` ID
directly: `--model mlx-community/your-model`.

## Methodology

### Real-Time Factor (RTF)

For STT / codec / VAD / LID / STS workloads operating on input audio:

```
RTF = input_audio_duration / processing_wall_time
```

Higher is better; RTF > 1 means the model processes audio faster than real
time (the threshold for usable streaming).

For TTS workloads (text in, audio out), RTF is inverted:

```
RTF = output_audio_duration / processing_wall_time
```

Same interpretation: > 1 means generation outpaces playback.

### Memory

`peakGPU` is captured via `MLX.GPU.peakMemory` after each fixture, after
explicit `MLX.GPU.resetPeakMemory()` calls at the start of a benchmark run.
`resident` is the process-level resident set size from `mach_task_basic_info`.

### WER / Semantic-WER / CER

Standard Wagner-Fischer edit distance (`Tests/Benchmarks/Utils/WERCalculator.swift`).
Semantic WER applies normalization (lowercase, strip punctuation, expand
common contractions, remove filler words) before scoring — catches
meaning-altering differences without being penalized by stylistic noise.
CER is the same algorithm at character granularity; useful as a complement
to WER on near-miss errors.

### SI-SNR (codec reconstruction)

Scale-invariant SNR computed against the reference input audio. Returns
nil when shapes don't align (sample-rate mismatch); otherwise expressed in
dB with higher-is-better.

## Skipping benchmarks in CI

The `MLXAudioBenchmarks` target is excluded from the default `xcodebuild
test` invocation in CI via `-skip-testing:MLXAudioBenchmarks`. Benchmarks
are operator-run and require model downloads that are too slow for PR
validation.

## Output anatomy

```
benchmarks/
├── m1-max-64gb-2026-05-06.md           ← rendered report
├── .m1-max-64gb-2026-05-06.state.json  ← persistent state (don't edit)
└── README.md                           ← this file
```

The state sidecar is the source of truth — the `.md` is regenerated from
it on every append. Hand-editing the markdown will be lost on the next run.
