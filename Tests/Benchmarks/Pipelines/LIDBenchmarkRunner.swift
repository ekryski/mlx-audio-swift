import Foundation
import MLX
import MLXAudioCore
import MLXAudioLID

/// Language-ID benchmark runner. Captures top-1 / top-3 accuracy when
/// fixtures supply a `referenceLanguage` (BCP-47 tag), plus RTF.
enum LIDBenchmarkRunner {

    @MainActor
    static func run(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        workload: String
    ) async throws {
        guard family.pipeline == .lid else { return }

        let bundle = Bundle.module
        let manifestURL = try resolveManifestURL(pipeline: .lid, bundle: bundle)
        let (manifest, baseDir) = try FixtureLoader.load(at: manifestURL)
        let samples = FixtureLoader.applyEnvFilters(manifest.samples)

        guard !samples.isEmpty else {
            print("[BENCH] no LID fixtures matched filter — nothing to do")
            return
        }
        if variant.repoId.isEmpty {
            print("[BENCH] skipped: variant has no repoId")
            return
        }

        print("[BENCH] LID: loading \(family.shortName) (\(variant.quantization)) from \(variant.repoId)")
        let baselineMem = Memory.activeMemory
        MLX.GPU.resetPeakMemory()

        let loadStart = Date()
        let predict: (MLXArray) -> (top1: String, top3: [String], processingSec: Double)
        do {
            predict = try await loadAndConfigure(family: family.shortName, repoId: variant.repoId)
        } catch {
            print("[BENCH] failed to load: \(error)")
            return
        }
        let loadDurationSec = Date().timeIntervalSince(loadStart)

        var hits1 = 0
        var hits3 = 0
        var scored = 0

        for sample in samples {
            let url: URL
            do {
                url = try FixtureLoader.resolveAudio(sample: sample, baseDir: baseDir, bundle: bundle, pipeline: .lid)
            } catch {
                print("[BENCH] skip \(sample.id): \(error)")
                continue
            }
            let (sampleRate, audio) = try loadAudioArray(from: url)
            let inputDurSec = Double(audio.shape[0]) / Double(sampleRate)

            var elapsedTotal: Double = 0
            var top1 = ""
            var top3: [String] = []
            for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                let r = predict(audio)
                elapsedTotal += r.processingSec
                top1 = r.top1
                top3 = r.top3
            }
            let avgElapsed = elapsedTotal / Double(max(1, BenchmarkEnv.timedRuns))
            let rtf = avgElapsed > 0 ? inputDurSec / avgElapsed : 0

            var top1Acc: Double? = nil
            var top3Acc: Double? = nil
            if let expected = sample.referenceLanguage?.lowercased() {
                scored += 1
                if top1.lowercased() == expected { hits1 += 1 }
                if top3.contains(where: { $0.lowercased() == expected }) { hits3 += 1 }
                top1Acc = top1.lowercased() == expected ? 1.0 : 0.0
                top3Acc = top3.contains(where: { $0.lowercased() == expected }) ? 1.0 : 0.0
            }

            let result = BenchmarkWriter.Result(
                pipeline: .lid,
                workload: workload,
                fixture: sample.id,
                inputDurationSec: inputDurSec,
                processingTimeSec: avgElapsed,
                realTimeFactor: rtf,
                top1Accuracy: top1Acc,
                top3Accuracy: top3Acc,
                baselineGPU: baselineMem,
                peakGPU: Memory.peakMemory,
                residentMB: residentMB(),
                outputPreview: "top1=\(top1) top3=\(top3.prefix(3).joined(separator: ","))"
            )

            BenchmarkWriter.append(
                model: family.name,
                repoId: variant.repoId,
                quantization: variant.quantization,
                result: result,
                parameters: parametersForReport(family: family, variant: variant, loadDurationSec: loadDurationSec)
            )

            print("[RESULT] \(sample.id) top1=\(top1) RTF=\(String(format: "%.2f", rtf))")
        }

        if scored > 0 {
            print("[BENCH] aggregate top1=\(String(format: "%.3f", Double(hits1)/Double(scored))) top3=\(String(format: "%.3f", Double(hits3)/Double(scored))) over \(scored) labeled samples")
        }
        print("[MEM] peak=\(BenchmarkWriter.formatBytes(Memory.peakMemory))")
    }

    @MainActor
    private static func loadAndConfigure(
        family: String,
        repoId: String
    ) async throws -> (MLXArray) -> (top1: String, top3: [String], processingSec: Double) {
        switch family {
        case "mms-lid-256":
            let model = try await Wav2Vec2ForSequenceClassification.fromPretrained(repoId)
            return { audio in
                let start = Date()
                let out = model.predict(waveform: audio, topK: 3)
                let elapsed = Date().timeIntervalSince(start)
                return (out.language, out.topLanguages.map { $0.language }, elapsed)
            }
        default:
            throw NSError(domain: "LIDBenchmarkRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No LID loader for '\(family)'."])
        }
    }

    private static func resolveManifestURL(pipeline: ModelRegistry.Pipeline, bundle: Bundle) throws -> URL {
        if let override = BenchmarkEnv.manifestPath {
            return URL(fileURLWithPath: override)
        }
        let subdir = "Resources/\(pipeline.rawValue)"
        if let url = bundle.url(forResource: "manifest", withExtension: "json", subdirectory: subdir) {
            return url
        }
        let projectRoot = BenchmarkWriter.projectRoot()
        return projectRoot.appendingPathComponent("Tests/Benchmarks/\(subdir)/manifest.json")
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
            ("Timed runs", "\(BenchmarkEnv.timedRuns)"),
        ]
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
