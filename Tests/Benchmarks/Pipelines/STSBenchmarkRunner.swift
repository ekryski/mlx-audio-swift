import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTS

/// Speech-to-speech benchmark runner. STS covers a heterogeneous mix —
/// enhancement (DeepFilterNet, MossFormer2-SE), translation (SAM-Audio), and
/// understanding (LFM2.5-Audio). The metric set differs per sub-task:
///
///   - enhancement:   SI-SNR vs reference clean audio
///   - translation:   BLEU vs reference translation
///   - understanding: RTF + token-throughput
///
/// First-cut here wires up only LFM2.5-Audio in text-to-text mode (the
/// minimal end-to-end demonstration). Enhancement-side scoring can reuse
/// `CodecBenchmarkRunner.computeSISNR` once we add a DeepFilterNet loader.
enum STSBenchmarkRunner {

    @MainActor
    static func run(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        workload: String
    ) async throws {
        guard family.pipeline == .sts else { return }

        let bundle = Bundle.module
        let manifestURL = try resolveManifestURL(pipeline: .sts, bundle: bundle)
        let (manifest, _) = try FixtureLoader.load(at: manifestURL)
        let samples = FixtureLoader.applyEnvFilters(manifest.samples)

        guard !samples.isEmpty else {
            print("[BENCH] no STS fixtures matched filter — nothing to do")
            return
        }
        if variant.repoId.isEmpty {
            print("[BENCH] skipped: variant has no repoId")
            return
        }

        print("[BENCH] STS: loading \(family.shortName) (\(variant.quantization)) from \(variant.repoId)")
        let baselineMem = Memory.activeMemory
        MLX.GPU.resetPeakMemory()

        let loadStart = Date()
        let runOnce: (String) async throws -> (output: String, processingSec: Double)
        do {
            runOnce = try await loadAndConfigure(family: family.shortName, repoId: variant.repoId)
        } catch {
            print("[BENCH] failed to load: \(error)")
            return
        }
        let loadDurationSec = Date().timeIntervalSince(loadStart)

        for sample in samples {
            guard let text = sample.effectiveText, !text.isEmpty else {
                print("[BENCH] skip \(sample.id): no text")
                continue
            }

            var elapsedTotal: Double = 0
            var lastOutput = ""
            for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                let r = try await runOnce(text)
                elapsedTotal += r.processingSec
                lastOutput = r.output
            }
            let avgElapsed = elapsedTotal / Double(max(1, BenchmarkEnv.timedRuns))

            let result = BenchmarkWriter.Result(
                pipeline: .sts,
                workload: workload,
                fixture: sample.id,
                processingTimeSec: avgElapsed,
                realTimeFactor: nil,  // RTF is workload-specific; leave nil for text-mode
                baselineGPU: baselineMem,
                peakGPU: Memory.peakMemory,
                residentMB: residentMB(),
                outputPreview: lastOutput
            )

            BenchmarkWriter.append(
                model: family.name,
                repoId: variant.repoId,
                quantization: variant.quantization,
                result: result,
                parameters: parametersForReport(family: family, variant: variant, loadDurationSec: loadDurationSec)
            )

            print("[RESULT] \(sample.id) elapsed=\(String(format: "%.3f", avgElapsed))s output=\"\(lastOutput.prefix(60))\"")
        }

        print("[MEM] peak=\(BenchmarkWriter.formatBytes(Memory.peakMemory))")
    }

    @MainActor
    private static func loadAndConfigure(
        family: String,
        repoId: String
    ) async throws -> (String) async throws -> (output: String, processingSec: Double) {
        switch family {
        case "lfm25-audio-1.5b":
            let model = try await LFM2AudioModel.fromPretrained(repoId)
            guard let processor = model.processor else {
                throw NSError(domain: "STSBenchmarkRunner", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "LFM2 model missing processor"])
            }
            return { text in
                let chat = ChatState(processor: processor)
                chat.newTurn(role: "system")
                chat.addText("Answer briefly in one sentence.")
                chat.endTurn()
                chat.newTurn(role: "user")
                chat.addText(text)
                chat.endTurn()
                chat.newTurn(role: "assistant")

                let cfg = LFMGenerationConfig(maxNewTokens: 64, temperature: 0.8, topK: 50)
                let start = Date()
                var tokens: [Int] = []
                for try await (token, modality) in model.generateInterleaved(
                    textTokens: chat.getTextTokens(),
                    audioFeatures: chat.getAudioFeatures(),
                    modalities: chat.getModalities(),
                    config: cfg
                ) {
                    MLX.eval(token)
                    if modality == .text {
                        tokens.append(token.item(Int.self))
                    }
                }
                let elapsed = Date().timeIntervalSince(start)
                let decoded = processor.decodeText(tokens)
                return (decoded, elapsed)
            }
        default:
            throw NSError(domain: "STSBenchmarkRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No STS loader for '\(family)'."])
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
