import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT

/// STT benchmark runner — load a model, transcribe each fixture, compute
/// WER / CER / RTF / TTFW / words-per-second, and append a row per fixture
/// to the shared report.
///
/// The runner abstracts over the multiple STT model APIs in `MLXAudioSTT`
/// (Parakeet, Qwen3-ASR, GLMASR, …) via a closure-based dispatch. New
/// models are integrated by adding a case in `loadAndConfigure(...)`.
enum STTBenchmarkRunner {

    /// Pipeline-side description of one fixture's transcription result.
    struct TranscriptionResult {
        let text: String
        let durationSec: Double
        let firstTokenLatencyMs: Double?
    }

    /// Run the STT benchmark for the resolved (model, variant, workload).
    /// Reads fixtures from the manifest path in env, falling back to the
    /// bundled canonical corpus.
    @MainActor
    static func run(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        workload: String
    ) async throws {
        guard family.pipeline == .stt else { return }

        let bundle = Bundle.module
        let manifestURL = try resolveManifestURL(pipeline: .stt, bundle: bundle)
        let (manifest, baseDir) = try FixtureLoader.load(at: manifestURL)
        let samples = FixtureLoader.applyEnvFilters(manifest.samples)

        guard !samples.isEmpty else {
            print("[BENCH] no fixtures matched filter — nothing to do")
            return
        }

        print("[BENCH] STT: loading \(family.shortName) (\(variant.quantization)) from \(variant.repoId)")
        if variant.repoId.isEmpty {
            print("[BENCH] skipped: variant has no repoId")
            return
        }

        let baselineMem = Memory.activeMemory
        MLX.GPU.resetPeakMemory()
        let loadStart = Date()
        let transcriber: (MLXArray) throws -> TranscriptionResult
        do {
            transcriber = try await loadAndConfigure(repoId: variant.repoId, family: family.shortName)
        } catch {
            print("[BENCH] failed to load \(variant.repoId): \(error)")
            return
        }
        let loadDurationSec = Date().timeIntervalSince(loadStart)
        print("[BENCH] loaded in \(String(format: "%.2f", loadDurationSec))s")

        // Optional warmup pass.
        if BenchmarkEnv.warmupRuns > 0, let first = samples.first,
           let url = try? FixtureLoader.resolveAudio(sample: first, baseDir: baseDir, bundle: bundle, pipeline: .stt) {
            for _ in 0..<BenchmarkEnv.warmupRuns {
                if let (_, audio) = try? loadAudioArray(from: url) {
                    _ = try? transcriber(audio)
                }
            }
            print("[WARMUP] \(BenchmarkEnv.warmupRuns) warmup pass(es) complete")
        }

        for sample in samples {
            let url: URL
            do {
                url = try FixtureLoader.resolveAudio(sample: sample, baseDir: baseDir, bundle: bundle, pipeline: .stt)
            } catch {
                print("[BENCH] skip \(sample.id): \(error)")
                continue
            }

            let (sampleRate, audio) = try loadAudioArray(from: url)
            let inputDurSec = Double(audio.shape[0]) / Double(sampleRate)

            // Timed runs (averaged).
            var results: [TranscriptionResult] = []
            var elapsedTotal: Double = 0
            for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                let start = Date()
                let r = try transcriber(audio)
                elapsedTotal += Date().timeIntervalSince(start)
                results.append(r)
            }
            let runs = Double(results.count)
            let avgElapsed = elapsedTotal / runs
            let avgTTFW = avgOrNil(results.compactMap { $0.firstTokenLatencyMs })

            // Use the last hypothesis as the canonical one for scoring.
            let hypothesis = results.last?.text ?? ""
            let reference = sample.referenceText ?? ""

            let werResult = WERCalculator.wer(reference: reference, hypothesis: hypothesis)
            let semanticResult = WERCalculator.semanticWER(reference: reference, hypothesis: hypothesis)
            let cerResult = WERCalculator.cer(reference: reference, hypothesis: hypothesis)

            let words = hypothesis.split(separator: " ").count
            let wordsPerSec = avgElapsed > 0 ? Double(words) / avgElapsed : 0
            let rtf = inputDurSec > 0 ? avgElapsed / inputDurSec : 0

            let peak = Memory.peakMemory
            let resident = residentMB()

            let result = BenchmarkWriter.Result(
                pipeline: .stt,
                workload: workload,
                fixture: sample.id,
                batchSize: BenchmarkEnv.batchSize,
                inputDurationSec: inputDurSec,
                processingTimeSec: avgElapsed,
                realTimeFactor: rtf,
                ttftMs: avgTTFW,
                wer: werResult.rate,
                semanticWER: semanticResult.rate,
                cer: cerResult.rate,
                wordsPerSec: wordsPerSec,
                baselineGPU: baselineMem,
                peakGPU: peak,
                residentMB: resident,
                outputPreview: hypothesis
            )

            BenchmarkWriter.append(
                model: family.name,
                repoId: variant.repoId,
                quantization: variant.quantization,
                kvConfig: BenchmarkEnv.kvConfig ?? "default",
                result: result,
                parameters: parametersForReport(family: family, variant: variant, loadDurationSec: loadDurationSec)
            )

            print("[RESULT] \(sample.id) WER=\(String(format: "%.4f", werResult.rate)) RTF=\(String(format: "%.2f", rtf)) elapsed=\(String(format: "%.3f", avgElapsed))s")
        }

        print("[MEM] peak=\(BenchmarkWriter.formatBytes(Memory.peakMemory))")
    }

    // MARK: - Model dispatch

    /// Load a model and return a closure that runs a single transcription
    /// pass. The closure normalizes return values into `TranscriptionResult`.
    @MainActor
    static func loadAndConfigure(repoId: String, family: String) async throws -> (MLXArray) throws -> TranscriptionResult {
        // Each STT model in MLXAudioSTT exposes `fromPretrained(_:)` async +
        // a `generate(audio:)` (or similar) method. The closure dispatch
        // below stays narrow on purpose: it's the only place per-model API
        // differences leak in. New families add a new branch.
        switch family {
        case "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3":
            let model = try await ParakeetModel.fromPretrained(repoId)
            return { audio in
                let out = model.generate(audio: audio, generationParameters: STTGenerateParameters())
                return TranscriptionResult(text: out.text, durationSec: 0, firstTokenLatencyMs: nil)
            }
        case "qwen3-asr-0.6b", "qwen3-asr-1.7b":
            let model = try await Qwen3ASRModel.fromPretrained(repoId)
            return { audio in
                let out = model.generate(audio: audio)
                return TranscriptionResult(text: out.text, durationSec: 0, firstTokenLatencyMs: nil)
            }
        case "glm-asr-nano":
            let model = try await GLMASRModel.fromPretrained(repoId)
            return { audio in
                let out = model.generate(audio: audio)
                return TranscriptionResult(text: out.text, durationSec: 0, firstTokenLatencyMs: nil)
            }
        case "granite-speech-1b":
            let model = try await GraniteSpeechModel.fromPretrained(repoId)
            return { audio in
                let out = model.generate(audio: audio)
                return TranscriptionResult(text: out.text, durationSec: 0, firstTokenLatencyMs: nil)
            }
        default:
            throw NSError(
                domain: "STTBenchmarkRunner",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No STT loader registered for family '\(family)'. Add a case in loadAndConfigure(...)."]
            )
        }
    }

    // MARK: - Helpers

    private static func resolveManifestURL(pipeline: ModelRegistry.Pipeline, bundle: Bundle) throws -> URL {
        if let override = BenchmarkEnv.manifestPath {
            return URL(fileURLWithPath: override)
        }
        let subdir = "Resources/\(pipeline.rawValue)/canonical"
        if let url = bundle.url(forResource: "manifest", withExtension: "json", subdirectory: subdir) {
            return url
        }
        // Fallback: `Tests/Benchmarks/Resources/{pipeline}/canonical/manifest.json`
        // when running outside an .xctest bundle.
        let projectRoot = BenchmarkWriter.projectRoot()
        return projectRoot.appendingPathComponent("Tests/Benchmarks/\(subdir)/manifest.json")
    }

    private static func avgOrNil(_ xs: [Double]) -> Double? {
        xs.isEmpty ? nil : xs.reduce(0, +) / Double(xs.count)
    }

    private static func parametersForReport(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        loadDurationSec: Double
    ) -> [(String, String)] {
        var rows: [(String, String)] = [
            ("Repo", variant.repoId),
            ("Quantization", variant.quantization),
            ("Load time", String(format: "%.2fs", loadDurationSec)),
            ("Warmup runs", "\(BenchmarkEnv.warmupRuns)"),
            ("Timed runs", "\(BenchmarkEnv.timedRuns)"),
            ("Batch size", "\(BenchmarkEnv.batchSize)"),
        ]
        if let lang = BenchmarkEnv.language {
            rows.append(("Language", lang))
        }
        if let notes = family.notes, !notes.isEmpty {
            rows.append(("Notes", notes))
        }
        return rows
    }

    private static func residentMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576 : 0
    }
}
