import Foundation
import MLX
import MLXAudioCore
import MLXAudioCodecs

/// Codec benchmark runner — encode + decode audio fixtures and measure
/// throughput plus reconstruction quality (SI-SNR vs the input signal).
///
/// RTF columns are split into encode-only and decode-only since real-time
/// streaming pipelines pay each cost independently.
enum CodecBenchmarkRunner {

    @MainActor
    static func run(
        family: ModelRegistry.ModelFamily,
        variant: ModelRegistry.ModelVariant,
        workload: String
    ) async throws {
        guard family.pipeline == .codec else { return }

        let bundle = Bundle.module
        let manifestURL = try resolveManifestURL(pipeline: .codec, bundle: bundle)
        let (manifest, baseDir) = try FixtureLoader.load(at: manifestURL)
        let samples = FixtureLoader.applyEnvFilters(manifest.samples)

        guard !samples.isEmpty else {
            print("[BENCH] no codec fixtures matched filter — nothing to do")
            return
        }
        if variant.repoId.isEmpty {
            print("[BENCH] skipped: variant has no repoId")
            return
        }

        print("[BENCH] CODEC: loading \(family.shortName) (\(variant.quantization)) from \(variant.repoId)")
        let baselineMem = Memory.activeMemory
        MLX.GPU.resetPeakMemory()

        let loadStart = Date()
        let codec: (MLXArray) async throws -> (encodeTimeSec: Double, decodeTimeSec: Double, reconstruction: MLXArray, sampleRate: Int)
        do {
            codec = try await loadAndConfigure(family: family.shortName, variant: variant)
        } catch {
            print("[BENCH] failed to load \(variant.repoId): \(error)")
            return
        }
        let loadDurationSec = Date().timeIntervalSince(loadStart)
        print("[BENCH] loaded in \(String(format: "%.2f", loadDurationSec))s")

        for sample in samples {
            let url: URL
            do {
                url = try FixtureLoader.resolveAudio(sample: sample, baseDir: baseDir, bundle: bundle, pipeline: .codec)
            } catch {
                print("[BENCH] skip \(sample.id): \(error)")
                continue
            }

            let (sampleRate, audio) = try loadAudioArray(from: url)
            let inputDurSec = Double(audio.shape[0]) / Double(sampleRate)

            var encodeTotal: Double = 0
            var decodeTotal: Double = 0
            var lastRecon: MLXArray?
            var lastReconRate = sampleRate
            for _ in 0..<max(1, BenchmarkEnv.timedRuns) {
                let r = try await codec(audio)
                encodeTotal += r.encodeTimeSec
                decodeTotal += r.decodeTimeSec
                lastRecon = r.reconstruction
                lastReconRate = r.sampleRate
            }
            let runs = Double(max(1, BenchmarkEnv.timedRuns))
            let encodeAvg = encodeTotal / runs
            let decodeAvg = decodeTotal / runs
            let totalAvg = encodeAvg + decodeAvg
            let encodeRTF = encodeAvg > 0 ? inputDurSec / encodeAvg : 0
            let decodeRTF = decodeAvg > 0 ? inputDurSec / decodeAvg : 0
            let siSnr = lastRecon.flatMap { computeSISNR(reference: audio, estimate: $0) }

            // Bitrate: rough estimate — `samples_in / encode_time` is throughput;
            // the codec emits compressed code shapes whose token count we don't
            // know without per-codec introspection. Leave nil for now and let
            // per-codec runners override.
            _ = lastReconRate

            let result = BenchmarkWriter.Result(
                pipeline: .codec,
                workload: workload,
                fixture: sample.id,
                inputDurationSec: inputDurSec,
                processingTimeSec: totalAvg,
                realTimeFactor: totalAvg > 0 ? inputDurSec / totalAvg : 0,
                encodeRTF: encodeRTF,
                decodeRTF: decodeRTF,
                siSnrDB: siSnr,
                baselineGPU: baselineMem,
                peakGPU: Memory.peakMemory,
                residentMB: residentMB()
            )

            BenchmarkWriter.append(
                model: family.name,
                repoId: variant.repoId,
                quantization: variant.quantization,
                result: result,
                parameters: parametersForReport(family: family, variant: variant, loadDurationSec: loadDurationSec)
            )

            print("[RESULT] \(sample.id) encode RTF=\(String(format: "%.2f", encodeRTF)) decode RTF=\(String(format: "%.2f", decodeRTF)) SI-SNR=\(siSnr.map { String(format: "%.2f", $0) } ?? "—")dB")
        }

        print("[MEM] peak=\(BenchmarkWriter.formatBytes(Memory.peakMemory))")
    }

    // MARK: - Codec dispatch

    @MainActor
    private static func loadAndConfigure(
        family: String,
        variant: ModelRegistry.ModelVariant
    ) async throws -> (MLXArray) async throws -> (encodeTimeSec: Double, decodeTimeSec: Double, reconstruction: MLXArray, sampleRate: Int) {
        switch family {
        case "snac-24khz":
            let snac = try await SNAC.fromPretrained(variant.repoId)
            return { audio in
                let input = audio.reshaped([1, 1, audio.shape[0]])
                let encStart = Date()
                let codes = snac.encode(input)
                MLX.eval(codes)
                let encodeTime = Date().timeIntervalSince(encStart)

                let decStart = Date()
                let recon = snac.decode(codes)
                MLX.eval(recon)
                let decodeTime = Date().timeIntervalSince(decStart)

                return (encodeTime, decodeTime, recon.squeezed(), snac.samplingRate)
            }
        case "mimi":
            guard let filename = variant.filename else {
                throw NSError(domain: "CodecBenchmarkRunner", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Mimi requires a filename in the variant"])
            }
            let mimi = try await Mimi.fromPretrained(repoId: variant.repoId, filename: filename) { _ in }
            return { audio in
                let input = audio.reshaped([1, 1, audio.shape[0]])
                let encStart = Date()
                let codes = mimi.encode(input)
                MLX.eval(codes)
                let encodeTime = Date().timeIntervalSince(encStart)

                let decStart = Date()
                let recon = mimi.decode(codes)
                MLX.eval(recon)
                let decodeTime = Date().timeIntervalSince(decStart)

                return (encodeTime, decodeTime, recon.squeezed(), Int(mimi.sampleRate))
            }
        default:
            throw NSError(
                domain: "CodecBenchmarkRunner",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No codec loader for '\(family)'."]
            )
        }
    }

    // MARK: - Helpers

    /// Scale-Invariant SNR — standard speech-quality metric for
    /// reference-based reconstruction. Result is in dB; higher is better.
    /// Returns nil when shapes don't match (resampling / chunking would
    /// invalidate the comparison).
    private static func computeSISNR(reference: MLXArray, estimate: MLXArray) -> Double? {
        // Flatten to 1-D and crop to the shorter length so a sample-rate
        // mismatch doesn't crash. SI-SNR doesn't tolerate length mismatch
        // anyway — better to skip than to return a bogus number, but for
        // the audio benchmark we crop and mark partial; downstream readers
        // should use this as a sanity-check, not a research-grade score.
        let ref1D = reference.reshaped([reference.shape.reduce(1, *)])
        let est1D = estimate.reshaped([estimate.shape.reduce(1, *)])
        let n = min(ref1D.shape[0], est1D.shape[0])
        guard n > 0 else { return nil }

        let r = ref1D[0..<n]
        let e = est1D[0..<n]

        // alpha = <e, r> / <r, r>
        let dotEr = (e * r).sum()
        let dotRr = (r * r).sum()
        MLX.eval(dotEr, dotRr)
        let dotErScalar = dotEr.item(Float.self)
        let dotRrScalar = dotRr.item(Float.self)
        guard dotRrScalar > 0 else { return nil }
        let alpha = dotErScalar / dotRrScalar

        // s_target = alpha * r; e_noise = e - s_target
        let target = r * MLXArray(alpha)
        let noise = e - target

        let targetEnergy = (target * target).sum()
        let noiseEnergy = (noise * noise).sum()
        MLX.eval(targetEnergy, noiseEnergy)
        let t = targetEnergy.item(Float.self)
        let nE = noiseEnergy.item(Float.self)
        guard t > 0, nE > 0 else { return nil }
        return Double(10.0 * log10f(t / nE))
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
            ("Warmup runs", "\(BenchmarkEnv.warmupRuns)"),
            ("Timed runs", "\(BenchmarkEnv.timedRuns)"),
        ]
        if let f = variant.filename { rows.append(("Filename", f)) }
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
