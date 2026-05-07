import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

/// TTS benchmark runner — load a model, synthesize each text fixture, and
/// report RTF (output-audio / processing-time), chars-per-second,
/// samples-per-second, TTFA, peak GPU, and a sample preview of the input
/// text.
///
/// `audio_seconds` here is the *generated* audio duration (the metric the
/// user cares about), not the input text length. RTF > 1 means the model
/// generates faster than realtime — the threshold for usable streaming.
enum TTSBenchmarkRunner {

    struct SynthesisResult {
        let audio: MLXArray
        let sampleRate: Int
        let firstAudioLatencyMs: Double?
    }

    @MainActor
    static func run(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        workload: String
    ) async throws {
        guard family.pipeline == .tts else { return }

        let bundle = Bundle.module
        let manifestURL = try resolveManifestURL(pipeline: .tts, bundle: bundle)
        let (manifest, _) = try FixtureLoader.load(at: manifestURL)
        let samples = FixtureLoader.applyEnvFilters(manifest.samples)

        guard !samples.isEmpty else {
            print("[BENCH] no fixtures matched filter — nothing to do")
            return
        }

        print("[BENCH] TTS: loading \(family.shortName) (\(variant.quantization)) from \(variant.repoId)")
        if variant.repoId.isEmpty {
            print("[BENCH] skipped: variant has no repoId")
            return
        }

        let baselineMem = Memory.activeMemory
        MLX.GPU.resetPeakMemory()
        let loadStart = Date()
        let synth: (String, String?) async throws -> SynthesisResult
        do {
            synth = try await loadAndConfigure(repoId: variant.repoId, family: family.shortName)
        } catch {
            print("[BENCH] failed to load \(variant.repoId): \(error)")
            return
        }
        let loadDurationSec = Date().timeIntervalSince(loadStart)
        print("[BENCH] loaded in \(String(format: "%.2f", loadDurationSec))s")

        let voice = BenchmarkEnv.voice ?? "af_heart"

        // Warmup
        if BenchmarkEnv.warmupRuns > 0, let first = samples.first, let text = first.effectiveText {
            for _ in 0..<BenchmarkEnv.warmupRuns {
                _ = try? await synth(text, voice)
            }
            print("[WARMUP] \(BenchmarkEnv.warmupRuns) warmup pass(es) complete")
        }

        for sample in samples {
            guard let text = sample.effectiveText, !text.isEmpty else {
                print("[BENCH] skip \(sample.id): no text")
                continue
            }

            var elapsedTotal: Double = 0
            var lastResult: SynthesisResult?
            var ttfaSamples: [Double] = []

            for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                let start = Date()
                let result = try await synth(text, voice)
                elapsedTotal += Date().timeIntervalSince(start)
                lastResult = result
                if let ttfa = result.firstAudioLatencyMs { ttfaSamples.append(ttfa) }
            }

            guard let r = lastResult else { continue }
            let runs = Double(max(1, BenchmarkEnv.timedRuns))
            let avgElapsed = elapsedTotal / runs
            let outDurSec = Double(r.audio.shape[r.audio.ndim - 1]) / Double(r.sampleRate)
            let rtf = avgElapsed > 0 ? outDurSec / avgElapsed : 0
            let charsPerSec = avgElapsed > 0 ? Double(text.count) / avgElapsed : 0
            let samplesPerSec = avgElapsed > 0 ? Double(r.audio.shape[r.audio.ndim - 1]) / avgElapsed : 0
            let ttfa = ttfaSamples.isEmpty ? nil : ttfaSamples.reduce(0, +) / Double(ttfaSamples.count)

            let peak = Memory.peakMemory
            let resident = residentMB()

            let result = BenchmarkWriter.Result(
                pipeline: .tts,
                workload: workload,
                fixture: sample.id,
                inputCharacters: text.count,
                outputDurationSec: outDurSec,
                processingTimeSec: avgElapsed,
                realTimeFactor: rtf,
                ttftMs: ttfa,
                charsPerSec: charsPerSec,
                samplesPerSec: samplesPerSec,
                baselineGPU: baselineMem,
                peakGPU: peak,
                residentMB: resident,
                outputPreview: String(text.prefix(120))
            )

            BenchmarkWriter.append(
                model: family.name,
                repoId: variant.repoId,
                quantization: variant.quantization,
                kvConfig: BenchmarkEnv.kvConfig ?? "default",
                configKeyExtras: [("voice", voice)],
                result: result,
                parameters: parametersForReport(family: family, variant: variant, voice: voice, loadDurationSec: loadDurationSec)
            )

            print("[RESULT] \(sample.id) RTF=\(String(format: "%.2f", rtf)) chars/s=\(String(format: "%.1f", charsPerSec)) elapsed=\(String(format: "%.3f", avgElapsed))s")
        }

        print("[MEM] peak=\(BenchmarkWriter.formatBytes(Memory.peakMemory))")
    }

    // MARK: - Model dispatch

    @MainActor
    static func loadAndConfigure(repoId: String, family: String) async throws -> (String, String?) async throws -> SynthesisResult {
        switch family {
        case "kokoro-82m", "kitten-tts":
            let model = try await KokoroModel.fromPretrained(repoId)
            return { (text: String, voice: String?) in
                let audio = try await model.generate(
                    text: text,
                    voice: voice,
                    refAudio: nil,
                    refText: nil,
                    language: nil,
                    generationParameters: GenerateParameters()
                )
                return SynthesisResult(audio: audio, sampleRate: model.sampleRate, firstAudioLatencyMs: nil)
            }
        default:
            throw NSError(
                domain: "TTSBenchmarkRunner",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No TTS loader registered for family '\(family)'. Add a case in loadAndConfigure(...)."]
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
        let projectRoot = BenchmarkWriter.projectRoot()
        return projectRoot.appendingPathComponent("Tests/Benchmarks/\(subdir)/manifest.json")
    }

    private static func parametersForReport(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        voice: String,
        loadDurationSec: Double
    ) -> [(String, String)] {
        var rows: [(String, String)] = [
            ("Repo", variant.repoId),
            ("Quantization", variant.quantization),
            ("Voice", voice),
            ("Load time", String(format: "%.2fs", loadDurationSec)),
            ("Warmup runs", "\(BenchmarkEnv.warmupRuns)"),
            ("Timed runs", "\(BenchmarkEnv.timedRuns)"),
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
