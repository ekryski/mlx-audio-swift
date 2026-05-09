import Foundation
import MLX
import Testing

/// Entry point for the audio benchmark suite.
///
/// One `@Test` per pipeline. The shell driver
/// (`scripts/benchmark.sh`) sets `MLX_AUDIO_BENCH_PIPELINE` to one of
/// `stt | tts | codec | vad | lid | sts` and invokes
/// `swift test --skip-build -c release --filter "MLXAudioBenchmarks"` —
/// the matching test runs, the others bail out as no-ops.
///
/// Tests are `.serialized` because they hold a singleton MLX device + GPU
/// memory; running concurrently would interleave peak-memory readings.
///
/// Modelled on `mlx-swift-lm/Tests/Benchmarks/InferenceBenchmark.swift` —
/// same dispatch shape, same `[BENCH]/[RESULT]/[MEM]` log conventions so
/// the shell driver can grep both repos with the same regex.
@Suite("Audio Benchmarks", .serialized)
struct AudioBenchmarks {

    @Test @MainActor func benchmark() async throws {
        printBuildEnvironment()

        guard let pipeline = BenchmarkEnv.pipeline else {
            print("[BENCH] no MLX_AUDIO_BENCH_PIPELINE set — nothing to do")
            print("[BENCH] usage: see benchmarks/README.md or run scripts/benchmark.sh --help")
            return
        }
        guard let modelShortName = BenchmarkEnv.model else {
            print("[BENCH] no MLX_AUDIO_BENCH_MODEL set — nothing to do")
            return
        }

        // Resolve the family + variant.
        let registryPipeline = registryPipeline(for: pipeline)
        let family: ModelRegistry.ModelFamily
        if let registered = ModelRegistry.family(named: modelShortName, pipeline: registryPipeline) {
            family = registered
        } else {
            // Allow ad-hoc HuggingFace IDs via "owner/model".
            if modelShortName.contains("/") {
                family = ModelRegistry.customFamily(repoId: modelShortName, pipeline: registryPipeline)
                print("[BENCH] using custom family for repoId='\(modelShortName)'")
            } else {
                print("[BENCH] unknown model '\(modelShortName)' for pipeline '\(pipeline.rawValue)'")
                print("[BENCH] available: \(ModelRegistry.families(in: registryPipeline).map { $0.shortName }.joined(separator: ", "))")
                return
            }
        }
        let variant = family.resolveVariant(BenchmarkEnv.quantization)

        let workload = BenchmarkEnv.workload ?? defaultWorkload(for: pipeline)

        print("[ENV] pipeline=\(pipeline.rawValue) model=\(family.shortName) quant=\(variant.quantization) workload=\(workload)")
        print("[ENV] repoId=\(variant.repoId)")

        switch pipeline {
        case .stt:
            try await STTBenchmarkRunner.run(family: family, variant: variant, workload: workload)
        case .tts:
            try await TTSBenchmarkRunner.run(family: family, variant: variant, workload: workload)
        case .codec:
            try await CodecBenchmarkRunner.run(family: family, variant: variant, workload: workload)
        case .vad:
            try await VADBenchmarkRunner.run(family: family, variant: variant, workload: workload)
        case .lid:
            try await LIDBenchmarkRunner.run(family: family, variant: variant, workload: workload)
        case .sts:
            try await STSBenchmarkRunner.run(family: family, variant: variant, workload: workload)
        }
    }

    private func printBuildEnvironment() {
        SystemInfo.printHardwareInfo()
        let info = GPU.deviceInfo()
        print("[BENCH] GPU arch: \(info.architecture)")
        #if DEBUG
        print("[BENCH] WARNING: built in DEBUG mode — timings will be misleading. Re-run with `make build-tests` (release).")
        #else
        print("[BENCH] Build: release")
        #endif
    }

    private func registryPipeline(for env: BenchmarkEnv.Pipeline) -> ModelRegistry.Pipeline {
        switch env {
        case .stt:   return .stt
        case .tts:   return .tts
        case .codec: return .codec
        case .vad:   return .vad
        case .lid:   return .lid
        case .sts:   return .sts
        }
    }

    private func defaultWorkload(for pipeline: BenchmarkEnv.Pipeline) -> String {
        switch pipeline {
        case .stt:   return "transcription"
        case .tts:   return "synthesis"
        case .codec: return "encode-decode"
        case .vad:   return "diarization"
        case .lid:   return "classify"
        case .sts:   return "understand"
        }
    }
}
