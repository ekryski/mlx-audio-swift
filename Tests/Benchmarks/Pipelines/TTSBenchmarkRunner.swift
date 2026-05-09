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
///
/// The runner dispatches through `TTS.loadModel(modelRepo:)`, the unified
/// factory in `TTSModel.swift`. Adding a new TTS family requires no change
/// here — drop a registry entry and ensure the factory's switch handles
/// the model_type. TTFA is captured by consuming the model's
/// `generateStream(...)` and timing from the first emitted `.audio` event.
enum TTSBenchmarkRunner {

    @MainActor
    static func run(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        workload: String
    ) async throws {
        guard family.pipeline == .tts else { return }

        let bundle = Bundle.module
        let manifestURL = try resolveManifestURL(pipeline: .tts, bundle: bundle)
        let (manifest, baseDir) = try FixtureLoader.load(at: manifestURL)
        let samples = FixtureLoader.applyEnvFilters(manifest.samples)

        guard !samples.isEmpty else {
            print("[BENCH] no fixtures matched filter — nothing to do")
            return
        }

        if variant.repoId.isEmpty {
            print("[BENCH] skipped: variant has no repoId")
            return
        }

        print("[BENCH] TTS: loading \(family.shortName) (\(variant.quantization)) from \(variant.repoId)")
        let baselineMem = Memory.activeMemory
        MLX.GPU.resetPeakMemory()
        let loadStart = Date()
        let model: SpeechGenerationModel
        do {
            model = try await TTS.loadModel(modelRepo: variant.repoId)
        } catch {
            print("[BENCH] failed to load \(variant.repoId): \(error)")
            return
        }
        let loadDurationSec = Date().timeIntervalSince(loadStart)
        print("[BENCH] loaded in \(String(format: "%.2f", loadDurationSec))s")

        let voice = BenchmarkEnv.voice
        let refAudio = try resolveReferenceAudio(family: family, baseDir: baseDir, bundle: bundle)
        let refText = BenchmarkEnv.refText

        // Warmup
        if BenchmarkEnv.warmupRuns > 0, let first = samples.first, let text = first.effectiveText {
            for _ in 0..<BenchmarkEnv.warmupRuns {
                _ = try? await model.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: BenchmarkEnv.language,
                    generationParameters: nil
                )
            }
            print("[WARMUP] \(BenchmarkEnv.warmupRuns) warmup pass(es) complete")
        }

        for sample in samples {
            guard let text = sample.effectiveText, !text.isEmpty else {
                print("[BENCH] skip \(sample.id): no text")
                continue
            }

            var elapsedTotal: Double = 0
            var lastAudio: MLXArray?
            var ttfaSamples: [Double] = []

            for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                let runStart = Date()
                // Drive `generateStream` so we can capture TTFA (time to
                // first audio chunk). Inlined here — extracting into a
                // helper trips Swift 6 strict-concurrency because the
                // SpeechGenerationModel existential isn't Sendable and
                // the helper would have to send `model` across an async
                // boundary. Inside the @MainActor `run` we stay on-actor
                // throughout, which compiles cleanly.
                let stream = model.generateStream(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: BenchmarkEnv.language,
                    generationParameters: model.defaultGenerationParameters
                )
                var firstAudioMs: Double? = nil
                var chunks: [MLXArray] = []
                for try await event in stream {
                    if case .audio(let chunk) = event {
                        if firstAudioMs == nil {
                            firstAudioMs = Date().timeIntervalSince(runStart) * 1000.0
                        }
                        chunks.append(chunk)
                    }
                }

                let runAudio: MLXArray
                if chunks.isEmpty {
                    // Streaming yielded nothing — fall back to one-shot
                    // `generate` so we still capture audio + RTF, just
                    // without TTFA.
                    runAudio = try await model.generate(
                        text: text,
                        voice: voice,
                        refAudio: refAudio,
                        refText: refText,
                        language: BenchmarkEnv.language,
                        generationParameters: nil
                    )
                } else if chunks.count == 1 {
                    runAudio = chunks[0]
                } else {
                    let lastAxis = chunks[0].ndim - 1
                    runAudio = MLX.concatenated(chunks, axis: lastAxis)
                }

                elapsedTotal += Date().timeIntervalSince(runStart)
                lastAudio = runAudio
                if let ttfa = firstAudioMs { ttfaSamples.append(ttfa) }
            }

            guard let audio = lastAudio else { continue }
            let runs = Double(max(1, BenchmarkEnv.timedRuns))
            let avgElapsed = elapsedTotal / runs
            let outDurSec = Double(audio.shape[audio.ndim - 1]) / Double(model.sampleRate)
            let rtf = avgElapsed > 0 ? outDurSec / avgElapsed : 0
            let charsPerSec = avgElapsed > 0 ? Double(text.count) / avgElapsed : 0
            let samplesPerSec = avgElapsed > 0 ? Double(audio.shape[audio.ndim - 1]) / avgElapsed : 0
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
                configKeyExtras: configKeyExtras(voice: voice),
                result: result,
                parameters: parametersForReport(family: family, variant: variant, voice: voice, loadDurationSec: loadDurationSec)
            )

            let ttfaStr = ttfa.map { String(format: " TTFA=%.0fms", $0) } ?? ""
            print("[RESULT] \(sample.id) RTF=\(String(format: "%.2f", rtf)) chars/s=\(String(format: "%.1f", charsPerSec))\(ttfaStr) elapsed=\(String(format: "%.3f", avgElapsed))s")
        }

        print("[MEM] peak=\(BenchmarkWriter.formatBytes(Memory.peakMemory))")
    }

    // MARK: - Reference audio (voice cloning)

    /// Voice-cloning families (Chatterbox, Marvis) require a reference
    /// audio clip. The runner looks for an env override
    /// (`MLX_AUDIO_BENCH_REF_AUDIO`) first, then falls back to the
    /// canonical TTS reference-outputs directory.
    @MainActor
    private static func resolveReferenceAudio(
        family: ModelRegistry.ModelFamily,
        baseDir: URL,
        bundle: Bundle
    ) throws -> MLXArray? {
        guard family.requiresReferenceAudio else { return nil }
        if let path = BenchmarkEnv.refAudioPath {
            let url = URL(fileURLWithPath: path)
            let (_, audio) = try loadAudioArray(from: url)
            return audio
        }
        // Fallback — pick the first .wav under reference-outputs/ if
        // present, else the first canonical fixture.
        let candidates = [
            baseDir.deletingLastPathComponent().appendingPathComponent("reference-outputs"),
            baseDir
        ]
        for dir in candidates {
            if let entries = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) {
                if let wav = entries.first(where: { $0.pathExtension.lowercased() == "wav" }) {
                    let (_, audio) = try loadAudioArray(from: wav)
                    return audio
                }
            }
        }
        if let url = bundle.url(forResource: "conv_001", withExtension: "wav", subdirectory: "Resources/tts/reference-outputs") {
            let (_, audio) = try loadAudioArray(from: url)
            return audio
        }
        return nil
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

    private static func configKeyExtras(voice: String?) -> [(String, String)] {
        guard let voice else { return [] }
        return [("voice", voice)]
    }

    private static func parametersForReport(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        voice: String?,
        loadDurationSec: Double
    ) -> [(String, String)] {
        var rows: [(String, String)] = [
            ("Repo", variant.repoId),
            ("Quantization", variant.quantization),
            ("Load time", String(format: "%.2fs", loadDurationSec)),
            ("Warmup runs", "\(BenchmarkEnv.warmupRuns)"),
            ("Timed runs", "\(BenchmarkEnv.timedRuns)"),
        ]
        if let voice {
            rows.append(("Voice", voice))
        }
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
