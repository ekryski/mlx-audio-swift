import Foundation

/// Shared manifest format for benchmark fixtures across all pipelines.
///
/// Each fixture is a single sample with optional reference data depending on
/// the pipeline:
///   - STT:   `audioFile` + `referenceText`
///   - TTS:   `text`
///   - Codec: `audioFile`
///   - VAD:   `audioFile` + `referenceTurns` (JSON-encoded turn boundaries)
///   - LID:   `audioFile` + `referenceLanguage` (BCP-47)
///   - STS:   `audioFile` + optionally `referenceAudioFile` (for enhancement)
///
/// The manifest schema purposefully flattens all fields into one struct so
/// pipeline-agnostic loading code can stay simple. Runners read only the
/// fields they care about.
struct FixtureManifest: Codable {
    let version: String
    let samples: [FixtureSample]
}

struct FixtureSample: Codable {
    let id: String
    /// Path relative to the manifest's parent directory. Optional for
    /// text-only TTS samples.
    let audioFile: String?
    /// Reference text — STT ground truth or TTS input.
    let referenceText: String?
    /// Alias used by TTS manifests for clarity.
    let text: String?
    /// Domain tag — `clean`, `medical`, `coding`, etc. Mirrors Sam's schema.
    let domain: String?
    /// Acoustic / pipeline condition — `clean`, `noisy`, `streaming`, etc.
    let condition: String?
    let durationSeconds: Double?
    /// Provenance: `pipecat`, `kokoro-af-heart`, `echo-tts`, `recorded`, …
    /// Lets reports filter / weight by source.
    let source: String?
    /// VAD: JSON-encoded turn timestamps `[(start, end, speakerId)]`.
    let referenceTurns: String?
    /// LID: expected BCP-47 language tag.
    let referenceLanguage: String?
    /// STS: enhancement target audio.
    let referenceAudioFile: String?

    /// Effective input text — TTS manifests can use either `text` or
    /// `referenceText` interchangeably.
    var effectiveText: String? { text ?? referenceText }
}

enum FixtureLoader {

    enum LoadError: Error, CustomStringConvertible {
        case manifestNotFound(String)
        case decodeFailed(String, Error)
        case audioFileMissing(String)

        var description: String {
            switch self {
            case .manifestNotFound(let path):
                return "Manifest not found at \(path)"
            case .decodeFailed(let path, let err):
                return "Failed to decode manifest at \(path): \(err)"
            case .audioFileMissing(let path):
                return "Referenced audio file missing: \(path)"
            }
        }
    }

    /// Load a manifest from disk. The manifest path comes from
    /// `MLX_AUDIO_BENCH_MANIFEST` (operator override) or from the bundled
    /// canonical fixtures for this pipeline.
    static func load(at url: URL) throws -> (manifest: FixtureManifest, baseDir: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.manifestNotFound(url.path)
        }
        do {
            let data = try Data(contentsOf: url)
            let manifest = try JSONDecoder().decode(FixtureManifest.self, from: data)
            return (manifest, url.deletingLastPathComponent())
        } catch let decodeErr {
            throw LoadError.decodeFailed(url.path, decodeErr)
        }
    }

    /// Resolve an audio fixture relative to its manifest's directory, or
    /// fall back to the test bundle's `Resources/{pipeline}/canonical`
    /// directory.
    static func resolveAudio(
        sample: FixtureSample,
        baseDir: URL,
        bundle: Bundle?,
        pipeline: ModelRegistry.Pipeline
    ) throws -> URL {
        guard let audioFile = sample.audioFile else {
            throw LoadError.audioFileMissing("(no audioFile field for sample id=\(sample.id))")
        }

        // 1. Path relative to the manifest.
        let direct = baseDir.appendingPathComponent(audioFile)
        if FileManager.default.fileExists(atPath: direct.path) {
            return direct
        }

        // 2. Bundle fallback — `Resources/{pipeline}/canonical/{audioFile}`.
        if let bundle = bundle {
            let subdir = "Resources/\(pipeline.rawValue)/canonical"
            let stem = (audioFile as NSString).deletingPathExtension
            let ext = (audioFile as NSString).pathExtension
            if let url = bundle.url(forResource: stem, withExtension: ext, subdirectory: subdir) {
                return url
            }
        }

        throw LoadError.audioFileMissing(audioFile)
    }

    /// Apply optional fixture filters from env vars: `MLX_AUDIO_BENCH_SAMPLES`
    /// (id allowlist) and `MLX_AUDIO_BENCH_MAX_SAMPLES` (cap).
    static func applyEnvFilters(_ samples: [FixtureSample]) -> [FixtureSample] {
        var result = samples
        if let allowlist = BenchmarkEnv.sampleIDs, !allowlist.isEmpty {
            let set = Set(allowlist)
            result = result.filter { set.contains($0.id) }
        }
        if let cap = BenchmarkEnv.maxSamples, cap > 0, result.count > cap {
            result = Array(result.prefix(cap))
        }
        return result
    }
}
