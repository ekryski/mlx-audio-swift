import Foundation
import MLX

/// Writes audio benchmark results to a single hardware-dated markdown file
/// under `benchmarks/`, with a JSON sidecar so multiple `swift test`
/// invocations (one per permutation in the shell driver's sweep) accumulate
/// into the same report.
///
/// File naming: `{chip-slug}-{ram}gb-{YYYY-MM-DD}.md`
/// Example:     `m1-max-64gb-2026-05-06.md`
///
/// Each row carries a `pipeline` discriminator (stt / tts / codec / vad / lid
/// / sts) so the renderer can emit pipeline-appropriate columns. Models from
/// different pipelines never share a results table — each pipeline gets its
/// own section under the model's heading.
///
/// Modelled on `mlx-swift-lm/Tests/Benchmarks/Utils/BenchmarkWriter.swift`
/// (same chip detection, same idempotent-render-from-state pattern, same git
/// metadata embedding) so an operator running both LLM and audio benchmarks
/// gets visually consistent reports.
enum BenchmarkWriter {
    private static let lock = NSLock()

    // MARK: - Public append API

    /// Pipeline tag — drives column layout in the rendered markdown.
    enum Pipeline: String, Codable {
        case stt
        case tts
        case codec
        case vad
        case lid
        case sts
    }

    /// Single benchmark result. All fields are optional so the same struct
    /// covers every pipeline; the renderer ignores fields irrelevant to its
    /// column set.
    struct Result: Codable {
        // Required
        var pipeline: Pipeline
        var workload: String        // e.g., "transcription", "synthesis", "encode-decode"
        var fixture: String         // sample id or "n/a"

        // Common
        var batchSize: Int = 1
        var inputDurationSec: Double? = nil    // input audio length (STT/codec/VAD/LID/STS)
        var inputCharacters: Int? = nil        // input text length (TTS)
        var outputDurationSec: Double? = nil   // output audio length (TTS/codec/STS)
        var processingTimeSec: Double          // wall time for the timed run
        var realTimeFactor: Double? = nil      // RTF — defined per pipeline
        var ttftMs: Double? = nil              // time-to-first-token (STT) / time-to-first-audio (TTS)

        // STT-specific
        var wer: Double? = nil
        var semanticWER: Double? = nil
        var cer: Double? = nil
        var wordsPerSec: Double? = nil

        // TTS-specific
        var charsPerSec: Double? = nil
        var samplesPerSec: Double? = nil

        // Codec-specific
        var encodeRTF: Double? = nil
        var decodeRTF: Double? = nil
        var bitrateKbps: Double? = nil
        var siSnrDB: Double? = nil

        // VAD-specific
        var frameAccuracy: Double? = nil
        var falsePositiveRate: Double? = nil
        var falseNegativeRate: Double? = nil

        // LID-specific
        var top1Accuracy: Double? = nil
        var top3Accuracy: Double? = nil

        // Memory
        var baselineGPU: Int = 0
        var peakGPU: Int = 0
        var residentMB: Double = 0

        // Output preview (for diagnostic readability)
        var outputPreview: String? = nil
    }

    /// Append a result row. Rows are grouped by (model, pipeline, config),
    /// where config is `quantization / kvConfig / workload` plus any extra
    /// disambiguators in `configKeyExtras`.
    static func append(
        model: String,
        repoId: String = "",
        quantization: String,
        kvConfig: String = "default",
        configKeyExtras: [(String, String)] = [],
        result: Result,
        parameters: [(String, String)] = []
    ) {
        lock.lock()
        defer { lock.unlock() }

        let hw = hardwareInfo()
        let filename = sessionFilename(hw: hw)
        let projectRoot = Self.projectRoot()
        let markdownURL = projectRoot
            .appendingPathComponent("benchmarks")
            .appendingPathComponent(filename + ".md")
        let sidecarURL = projectRoot
            .appendingPathComponent("benchmarks")
            .appendingPathComponent("." + filename + ".state.json")

        try? FileManager.default.createDirectory(
            at: markdownURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var state = loadState(from: sidecarURL) ?? SessionState(
            chip: hw.chip,
            gpuArch: GPU.deviceInfo().architecture,
            systemRAM: hw.systemRAM,
            gpuLimit: hw.gpuLimit,
            osVersion: hw.osVersion,
            branch: gitBranch(),
            commit: gitLastCommitLine(),
            createdAt: isoDateTime(Date()),
            models: []
        )

        var configKey = "\(quantization) / \(kvConfig) / \(result.workload)"
        for (k, v) in configKeyExtras {
            configKey += " / \(k)=\(v)"
        }

        // Locate or create the (model, pipeline) bucket.
        if let mi = state.models.firstIndex(where: { $0.displayName == model }) {
            if let pi = state.models[mi].pipelines.firstIndex(where: { $0.pipeline == result.pipeline }) {
                if let ci = state.models[mi].pipelines[pi].configs.firstIndex(where: { $0.key == configKey }) {
                    state.models[mi].pipelines[pi].configs[ci].rows.append(result)
                } else {
                    state.models[mi].pipelines[pi].configs.append(ConfigEntry(
                        key: configKey,
                        quantization: quantization,
                        kvConfig: kvConfig,
                        workload: result.workload,
                        parameterRows: parameters.map { [$0.0, $0.1] },
                        rows: [result]
                    ))
                }
            } else {
                state.models[mi].pipelines.append(PipelineEntry(
                    pipeline: result.pipeline,
                    configs: [ConfigEntry(
                        key: configKey,
                        quantization: quantization,
                        kvConfig: kvConfig,
                        workload: result.workload,
                        parameterRows: parameters.map { [$0.0, $0.1] },
                        rows: [result]
                    )]
                ))
            }
        } else {
            state.models.append(ModelEntry(
                displayName: model,
                repoId: repoId,
                pipelines: [PipelineEntry(
                    pipeline: result.pipeline,
                    configs: [ConfigEntry(
                        key: configKey,
                        quantization: quantization,
                        kvConfig: kvConfig,
                        workload: result.workload,
                        parameterRows: parameters.map { [$0.0, $0.1] },
                        rows: [result]
                    )]
                )]
            ))
        }

        saveState(state, to: sidecarURL)
        let markdown = renderMarkdown(state: state)
        try? markdown.write(to: markdownURL, atomically: true, encoding: .utf8)
    }

    // MARK: - State types (JSON sidecar)

    private struct SessionState: Codable {
        var chip: String
        var gpuArch: String
        var systemRAM: String
        var gpuLimit: String
        var osVersion: String
        var branch: String
        var commit: String
        var createdAt: String
        var models: [ModelEntry]
    }

    private struct ModelEntry: Codable {
        var displayName: String
        var repoId: String
        var pipelines: [PipelineEntry]
    }

    private struct PipelineEntry: Codable {
        var pipeline: Pipeline
        var configs: [ConfigEntry]
    }

    private struct ConfigEntry: Codable {
        var key: String
        var quantization: String
        var kvConfig: String
        var workload: String
        var parameterRows: [[String]]
        var rows: [Result]
    }

    // MARK: - Sidecar I/O

    private static func loadState(from url: URL) -> SessionState? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SessionState.self, from: data)
    }

    private static func saveState(_ state: SessionState, to url: URL) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? enc.encode(state) else { return }
        try? data.write(to: url, options: .atomic)
    }

    // MARK: - Markdown rendering

    private static func renderMarkdown(state: SessionState) -> String {
        var md = ""
        md += "# Audio Benchmark: \(state.chip) — \(datePortion(state.createdAt))\n\n"
        md += "**Hardware:** \(state.chip), \(state.systemRAM) unified memory"
        md += " (GPU limit \(state.gpuLimit))\n"
        md += "**OS:** macOS \(state.osVersion)\n"
        md += "**Branch:** `\(state.branch)`\n"
        md += "**Commit:** \(state.commit)\n"
        md += "**Created:** \(state.createdAt)\n\n"

        if state.models.isEmpty {
            md += "_No benchmark rows recorded yet._\n"
        } else {
            md += "## Models\n\n"
            for m in state.models {
                md += renderModelSection(m)
            }
        }

        md += "## Methodology\n\n"
        md += "See [benchmarks/README.md](README.md#methodology) for column "
        md += "definitions, fixture sources, and how reports accumulate across "
        md += "permutations.\n"
        return md
    }

    private static func renderModelSection(_ model: ModelEntry) -> String {
        var md = "### \(model.displayName)\n\n"
        if !model.repoId.isEmpty {
            md += "**Model:** `\(model.repoId)`\n\n"
        }
        for pipeline in model.pipelines {
            md += renderPipelineSection(pipeline)
        }
        return md
    }

    private static func renderPipelineSection(_ entry: PipelineEntry) -> String {
        var md = "#### Pipeline: \(entry.pipeline.rawValue.uppercased())\n\n"
        md += pipelineTableHeader(entry.pipeline)
        for c in entry.configs {
            for r in c.rows {
                md += pipelineTableRow(c.key, result: r)
            }
        }
        md += "\n"

        if entry.configs.contains(where: { !$0.parameterRows.isEmpty }) {
            md += "##### Parameters\n\n"
            for c in entry.configs where !c.parameterRows.isEmpty {
                md += "**\(c.key)**\n\n"
                md += "| Parameter | Value |\n"
                md += "|-----------|-------|\n"
                for row in c.parameterRows where row.count == 2 {
                    md += "| \(row[0]) | \(row[1]) |\n"
                }
                md += "\n"
            }
        }
        return md
    }

    // MARK: - Per-pipeline table headers + rows

    private static func pipelineTableHeader(_ pipeline: Pipeline) -> String {
        switch pipeline {
        case .stt:
            return """
                | Config | Fixture | Audio (s) | WER | Sem-WER | CER | RTF | TTFW (ms) | Words/s | GPU Peak | Resident | Sample |
                |--------|---------|----------:|----:|--------:|----:|----:|----------:|--------:|---------:|---------:|--------|

                """
        case .tts:
            return """
                | Config | Fixture | Chars | Audio (s) | RTF | TTFA (ms) | Chars/s | Samples/s | GPU Peak | Resident | Sample |
                |--------|---------|------:|----------:|----:|----------:|--------:|----------:|---------:|---------:|--------|

                """
        case .codec:
            return """
                | Config | Fixture | Audio (s) | Encode RTF | Decode RTF | Bitrate kbps | SI-SNR dB | GPU Peak | Resident |
                |--------|---------|----------:|-----------:|-----------:|-------------:|----------:|---------:|---------:|

                """
        case .vad:
            return """
                | Config | Fixture | Audio (s) | RTF | Frame Acc | FPR | FNR | GPU Peak | Resident |
                |--------|---------|----------:|----:|----------:|----:|----:|---------:|---------:|

                """
        case .lid:
            return """
                | Config | Fixture | Audio (s) | RTF | Top-1 | Top-3 | GPU Peak | Resident |
                |--------|---------|----------:|----:|------:|------:|---------:|---------:|

                """
        case .sts:
            return """
                | Config | Fixture | Audio (s) | RTF | SI-SNR dB | GPU Peak | Resident | Sample |
                |--------|---------|----------:|----:|----------:|---------:|---------:|--------|

                """
        }
    }

    private static func pipelineTableRow(_ configKey: String, result r: Result) -> String {
        let cfg = mdTableCell(configKey)
        let fix = mdTableCell(r.fixture)
        let audio = r.inputDurationSec.map { String(format: "%.2f", $0) } ?? "—"
        let rtf = r.realTimeFactor.map { String(format: "%.2f", $0) } ?? "—"
        let gpu = formatBytes(r.peakGPU)
        let res = String(format: "%.0f MB", r.residentMB)

        switch r.pipeline {
        case .stt:
            return "| \(cfg) | \(fix) | \(audio) | \(fmt(r.wer)) | \(fmt(r.semanticWER)) | \(fmt(r.cer)) | \(rtf) | \(fmtMs(r.ttftMs)) | \(fmt1(r.wordsPerSec)) | \(gpu) | \(res) | \(formatSample(r.outputPreview)) |\n"
        case .tts:
            let chars = r.inputCharacters.map { "\($0)" } ?? "—"
            let outDur = r.outputDurationSec.map { String(format: "%.2f", $0) } ?? "—"
            return "| \(cfg) | \(fix) | \(chars) | \(outDur) | \(rtf) | \(fmtMs(r.ttftMs)) | \(fmt1(r.charsPerSec)) | \(fmt0(r.samplesPerSec)) | \(gpu) | \(res) | \(formatSample(r.outputPreview)) |\n"
        case .codec:
            return "| \(cfg) | \(fix) | \(audio) | \(fmt2(r.encodeRTF)) | \(fmt2(r.decodeRTF)) | \(fmt2(r.bitrateKbps)) | \(fmt2(r.siSnrDB)) | \(gpu) | \(res) |\n"
        case .vad:
            return "| \(cfg) | \(fix) | \(audio) | \(rtf) | \(fmt3(r.frameAccuracy)) | \(fmt3(r.falsePositiveRate)) | \(fmt3(r.falseNegativeRate)) | \(gpu) | \(res) |\n"
        case .lid:
            return "| \(cfg) | \(fix) | \(audio) | \(rtf) | \(fmt3(r.top1Accuracy)) | \(fmt3(r.top3Accuracy)) | \(gpu) | \(res) |\n"
        case .sts:
            return "| \(cfg) | \(fix) | \(audio) | \(rtf) | \(fmt2(r.siSnrDB)) | \(gpu) | \(res) | \(formatSample(r.outputPreview)) |\n"
        }
    }

    // MARK: - Format helpers

    private static func fmt(_ x: Double?) -> String { x.map { String(format: "%.4f", $0) } ?? "—" }
    private static func fmt1(_ x: Double?) -> String { x.map { String(format: "%.1f", $0) } ?? "—" }
    private static func fmt0(_ x: Double?) -> String { x.map { String(format: "%.0f", $0) } ?? "—" }
    private static func fmt2(_ x: Double?) -> String { x.map { String(format: "%.2f", $0) } ?? "—" }
    private static func fmt3(_ x: Double?) -> String { x.map { String(format: "%.3f", $0) } ?? "—" }
    private static func fmtMs(_ x: Double?) -> String { x.map { String(format: "%.0f", $0) } ?? "—" }

    private static func mdTableCell(_ s: String) -> String {
        s.replacingOccurrences(of: "|", with: "\\|")
    }

    private static func formatSample(_ preview: String?) -> String {
        guard let raw = preview, !raw.isEmpty else { return "—" }
        let oneLine = raw
            .replacingOccurrences(of: "\r\n", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
        let limit = 120
        let truncated = oneLine.count > limit
            ? String(oneLine.prefix(limit)) + "…"
            : oneLine
        return truncated
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "`", with: "'")
    }

    static func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1_073_741_824 {
            return String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
        }
        return String(format: "%.0f MB", Double(bytes) / 1_048_576)
    }

    // MARK: - Filename / session

    static func sessionFilename(hw: HardwareInfo) -> String {
        let chipSlug = chipSlug(from: hw.chip)
        let ramSlug = hw.systemRAM.lowercased()
        return "\(chipSlug)-\(ramSlug)-\(sessionDatePortion)"
    }

    private static func chipSlug(from chip: String) -> String {
        var s = chip
        if let paren = s.firstIndex(of: "(") {
            s = String(s[..<paren])
        }
        s = s.trimmingCharacters(in: .whitespaces).lowercased()
        if s.hasPrefix("apple ") { s = String(s.dropFirst("apple ".count)) }
        return s.split(separator: " ").joined(separator: "-")
    }

    nonisolated(unsafe) private static var _sessionDate: String?
    static var sessionDatePortion: String {
        if let cached = _sessionDate { return cached }
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        fmt.timeZone = .current
        let s = fmt.string(from: Date())
        _sessionDate = s
        return s
    }

    private static func datePortion(_ iso: String) -> String { String(iso.prefix(10)) }

    private static func isoDateTime(_ d: Date) -> String {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime]
        return fmt.string(from: d)
    }

    // MARK: - Project root / git

    static func projectRoot() -> URL {
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<5 {
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("Package.swift").path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private static func gitBranch() -> String {
        let root = projectRoot()
        let headPath = root.appendingPathComponent(".git/HEAD").path
        return (try? String(contentsOfFile: headPath, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "ref: refs/heads/", with: "") ?? "unknown"
    }

    private static func gitLastCommitLine() -> String {
        let root = projectRoot()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", root.path, "log", "-1", "--format=%h %s"]
        let out = Pipe()
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
            let data = out.fileHandleForReading.readDataToEndOfFile()
            guard process.terminationStatus == 0,
                  var line = String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                  !line.isEmpty
            else { return "`unknown`" }
            line = line.replacingOccurrences(of: "`", with: "'")
            line = line.replacingOccurrences(of: "|", with: "\\|")
            return "`\(line)`"
        } catch {
            return "`unknown`"
        }
    }

    // MARK: - Hardware

    struct HardwareInfo {
        let chip: String
        let systemRAM: String
        let gpuLimit: String
        let osVersion: String
    }

    static func hardwareInfo() -> HardwareInfo {
        let info = GPU.deviceInfo()
        let ramGB = Double(info.memorySize) / 1_073_741_824
        let gpuGB = Double(info.maxRecommendedWorkingSetSize) / 1_073_741_824
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let chipName = humanReadableChipName(gpuArch: info.architecture)
        return HardwareInfo(
            chip: chipName,
            systemRAM: String(format: "%.0fGB", ramGB),
            gpuLimit: String(format: "%.0fGB", gpuGB),
            osVersion: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
        )
    }

    private static func humanReadableChipName(gpuArch: String) -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var brand = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        let brandStr = decodeCString(brand)
        if !brandStr.isEmpty {
            return "\(brandStr) (\(gpuArch))"
        }
        size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let modelStr = decodeCString(model)
        return "\(appleChipName(from: modelStr, gpuArch: gpuArch)) (\(gpuArch))"
    }

    private static func decodeCString(_ array: [CChar]) -> String {
        let bytes = array.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func appleChipName(from model: String, gpuArch: String) -> String {
        let archLower = gpuArch.lowercased()
        let gen: String
        if archLower.contains("g13") { gen = "M1" }
        else if archLower.contains("g14") { gen = "M2" }
        else if archLower.contains("g15") { gen = "M3" }
        else if archLower.contains("g16") { gen = "M4" }
        else if archLower.contains("g17") { gen = "M5" }
        else { return model.isEmpty ? gpuArch : model }

        let variant: String
        if archLower.hasSuffix("c") { variant = "Ultra" }
        else if archLower.hasSuffix("x") { variant = "Max" }
        else if archLower.hasSuffix("d") { variant = "Pro" }
        else { variant = "" }

        return variant.isEmpty ? "Apple \(gen)" : "Apple \(gen) \(variant)"
    }
}
