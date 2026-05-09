import Foundation
import MLX
import MLXAudioCore
import MLXAudioVAD

/// VAD / diarization benchmark runner. Captures RTF + per-segment counts.
/// Frame-level accuracy + FPR/FNR scoring kicks in when fixtures supply
/// `referenceTurns` (left as a TODO until labeled fixtures are bundled).
enum VADBenchmarkRunner {

    @MainActor
    static func run(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        workload: String
    ) async throws {
        guard family.pipeline == .vad else { return }
        if variant.repoId.isEmpty {
            print("[BENCH] skipped: variant has no repoId")
            return
        }

        let bundle = Bundle.module
        let manifestURL = try resolveManifestURL(pipeline: .vad, bundle: bundle)
        let (manifest, baseDir) = try FixtureLoader.load(at: manifestURL)
        let samples = FixtureLoader.applyEnvFilters(manifest.samples)
        guard !samples.isEmpty else {
            print("[BENCH] no VAD fixtures matched filter — nothing to do")
            return
        }

        print("[BENCH] VAD: loading \(family.shortName) (\(variant.quantization)) from \(variant.repoId)")
        let baselineMem = Memory.activeMemory
        MLX.GPU.resetPeakMemory()

        let loadStart = Date()
        // Load + dispatch is inlined per family — passing the model out
        // through a closure trips Swift 6 strict-concurrency because the
        // model isn't Sendable. Each branch loads, then runs the per-fixture
        // loop directly, then returns. Add a new family by adding a case
        // here that follows the same shape.
        switch family.shortName {
        case "sortformer-streaming-4spk":
            let model: SortformerModel
            do {
                model = try await SortformerModel.fromPretrained(variant.repoId)
            } catch {
                print("[BENCH] failed to load \(variant.repoId): \(error)")
                return
            }
            let loadDurationSec = Date().timeIntervalSince(loadStart)
            print("[BENCH] loaded in \(String(format: "%.2f", loadDurationSec))s")
            for sample in samples {
                guard let url = try? FixtureLoader.resolveAudio(sample: sample, baseDir: baseDir, bundle: bundle, pipeline: .vad) else {
                    print("[BENCH] skip \(sample.id): audio missing")
                    continue
                }
                let (sampleRate, audio) = try loadAudioArray(from: url)
                let inputDurSec = Double(audio.shape[0]) / Double(sampleRate)

                var elapsedTotal: Double = 0
                var lastSegments = 0
                for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                    let runStart = Date()
                    let output = try await model.generate(audio: audio, verbose: false)
                    elapsedTotal += Date().timeIntervalSince(runStart)
                    lastSegments = output.segments.count
                }
                emitResult(family: family, variant: variant, sample: sample, inputDurSec: inputDurSec,
                           elapsedTotal: elapsedTotal, segments: lastSegments, workload: workload,
                           baselineMem: baselineMem, loadDurationSec: loadDurationSec)
            }

        case "silero-vad":
            let model: SileroVAD
            do {
                model = try await SileroVAD.fromPretrained(variant.repoId)
            } catch {
                print("[BENCH] failed to load \(variant.repoId): \(error)")
                return
            }
            let loadDurationSec = Date().timeIntervalSince(loadStart)
            print("[BENCH] loaded in \(String(format: "%.2f", loadDurationSec))s")
            for sample in samples {
                guard let url = try? FixtureLoader.resolveAudio(sample: sample, baseDir: baseDir, bundle: bundle, pipeline: .vad) else {
                    print("[BENCH] skip \(sample.id): audio missing")
                    continue
                }
                let (sampleRate, audio) = try loadAudioArray(from: url)
                let inputDurSec = Double(audio.shape[0]) / Double(sampleRate)

                var elapsedTotal: Double = 0
                var lastSegments = 0
                for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                    let runStart = Date()
                    let timestamps = try model.getSpeechTimestamps(audio, sampleRate: sampleRate)
                    elapsedTotal += Date().timeIntervalSince(runStart)
                    lastSegments = timestamps.count
                }
                emitResult(family: family, variant: variant, sample: sample, inputDurSec: inputDurSec,
                           elapsedTotal: elapsedTotal, segments: lastSegments, workload: workload,
                           baselineMem: baselineMem, loadDurationSec: loadDurationSec)
            }

        default:
            print("[BENCH] no VAD loader for '\(family.shortName)' — add a case in VADBenchmarkRunner.run(...)")
            return
        }

        print("[MEM] peak=\(BenchmarkWriter.formatBytes(Memory.peakMemory))")
    }

    @MainActor
    private static func emitResult(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        sample: FixtureSample,
        inputDurSec: Double,
        elapsedTotal: Double,
        segments: Int,
        workload: String,
        baselineMem: Int,
        loadDurationSec: Double
    ) {
        let avgElapsed = elapsedTotal / Double(max(1, BenchmarkEnv.timedRuns))
        let rtf = avgElapsed > 0 ? inputDurSec / avgElapsed : 0
        let result = BenchmarkWriter.Result(
            pipeline: .vad,
            workload: workload,
            fixture: sample.id,
            inputDurationSec: inputDurSec,
            processingTimeSec: avgElapsed,
            realTimeFactor: rtf,
            baselineGPU: baselineMem,
            peakGPU: Memory.peakMemory,
            residentMB: residentMB(),
            outputPreview: "\(segments) segments"
        )
        BenchmarkWriter.append(
            model: family.name,
            repoId: variant.repoId,
            quantization: variant.quantization,
            result: result,
            parameters: parametersForReport(family: family, variant: variant, loadDurationSec: loadDurationSec)
        )
        print("[RESULT] \(sample.id) RTF=\(String(format: "%.2f", rtf)) segments=\(segments)")
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
