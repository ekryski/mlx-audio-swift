import Foundation

/// Centralized accessor for `MLX_AUDIO_BENCH_*` environment variables that the
/// shell driver (`scripts/benchmark.sh`) sets per (model × quant × workload)
/// permutation.
///
/// The Swift test target reads these once per `swift test` invocation and
/// dispatches to the matching pipeline runner. Callers should treat unset
/// vars as "use the harness default" — never throw on a missing knob, because
/// the shell driver only exports vars that were explicitly requested by the
/// user.
enum BenchmarkEnv {

    private static func env(_ key: String) -> String? {
        ProcessInfo.processInfo.environment[key]
    }

    private static func envList(_ key: String) -> [String]? {
        env(key)?.split(separator: ",", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    // MARK: - Top-level dispatch

    /// Which pipeline benchmark to run. Empty / unset → no-op (so that
    /// `swift test --filter benchmark` without env doesn't error out, it
    /// just prints a usage hint).
    static var pipeline: Pipeline? {
        env("MLX_AUDIO_BENCH_PIPELINE").flatMap(Pipeline.init(rawValue:))
    }

    enum Pipeline: String {
        case stt
        case tts
        case codec
        case vad
        case lid
        case sts
    }

    // MARK: - Per-model dimensions

    /// Short-name from the registry (e.g., `parakeet-tdt-0.6b-v3`,
    /// `kokoro-82m`). Required for any non-no-op run.
    static var model: String? { env("MLX_AUDIO_BENCH_MODEL") }

    /// Variant key — `bf16`, `4bit`, `8bit`, etc. Resolves against the
    /// model's `ModelRegistry.ModelFamily.variants`.
    static var quantization: String? { env("MLX_AUDIO_BENCH_QUANT") }

    /// KV-cache strategy override (only applies to STT/TTS models that wrap
    /// an LLM-style decoder, e.g., Qwen3-ASR, Qwen3-TTS, Voxtral). Mirrors
    /// the dimension in `mlx-swift-lm`'s benchmark.
    static var kvConfig: String? { env("MLX_AUDIO_BENCH_KV") }

    // MARK: - Workload knobs

    /// Workload identifier, pipeline-specific:
    ///   STT:   transcription | batch | streaming | multilingual | noisy
    ///   TTS:   synthesis | streaming | longform | ttfa
    ///   Codec: encode-decode | streaming
    ///   VAD:   diarization | turn-detection
    ///   LID:   classify
    ///   STS:   enhance | translate | understand
    static var workload: String? { env("MLX_AUDIO_BENCH_WORKLOAD") }

    /// Optional language hint (BCP-47 or ISO-639). Models that auto-detect
    /// can ignore this; multilingual workloads use it to filter fixtures.
    static var language: String? { env("MLX_AUDIO_BENCH_LANGUAGE") }

    /// Batch size for STT. Defaults to 1 when unset.
    static var batchSize: Int { Int(env("MLX_AUDIO_BENCH_BATCH") ?? "") ?? 1 }

    /// Audio length cap in seconds for synthetic / streaming workloads. Some
    /// pipelines (codec, streaming-stt) loop until this duration is reached.
    static var audioLengthSeconds: Double? {
        env("MLX_AUDIO_BENCH_AUDIO_LEN").flatMap(Double.init)
    }

    /// Voice override for TTS workloads (e.g., `af_heart`, `af_bella`).
    /// When unset, runners pick a sensible default from the model.
    static var voice: String? { env("MLX_AUDIO_BENCH_VOICE") }

    /// Number of warmup runs before timed measurements. Defaults to 1.
    static var warmupRuns: Int { Int(env("MLX_AUDIO_BENCH_WARMUP") ?? "") ?? 1 }

    /// Number of timed runs to average over. Defaults to 1 — increase for
    /// noisy metrics, but each run is a full inference so cost scales
    /// linearly.
    static var timedRuns: Int { Int(env("MLX_AUDIO_BENCH_RUNS") ?? "") ?? 1 }

    /// Override the fixture manifest path. When unset, runners use the
    /// canonical bundle at `Resources/{pipeline}/canonical/manifest.json`.
    static var manifestPath: String? { env("MLX_AUDIO_BENCH_MANIFEST") }

    /// Optional per-sample limit. Useful for smoke runs.
    static var maxSamples: Int? {
        env("MLX_AUDIO_BENCH_MAX_SAMPLES").flatMap(Int.init)
    }

    /// Optional list filter — only fixtures whose `id` is in this list run.
    /// Comma-separated. Useful for re-running a single failing fixture.
    static var sampleIDs: [String]? { envList("MLX_AUDIO_BENCH_SAMPLES") }
}
