import Foundation
import MLX

/// Hardware detection and reporting for the audio benchmark suite. Mirrors
/// the helper in `mlx-swift-lm/Tests/Benchmarks/Utils/SystemInfo.swift` so
/// reports across the two repos can be cross-referenced (same chip slug,
/// same RAM/GPU-limit format).
enum SystemInfo {

    struct Hardware: CustomStringConvertible {
        let architecture: String
        let systemMemoryGB: Double
        let gpuMemoryLimitGB: Double

        var description: String {
            "\(architecture), \(String(format: "%.0f", systemMemoryGB))GB RAM, " +
                "\(String(format: "%.0f", gpuMemoryLimitGB))GB GPU limit"
        }
    }

    /// Query current hardware info via Metal / sysctl.
    static func hardware() -> Hardware {
        let info = GPU.deviceInfo()
        return Hardware(
            architecture: info.architecture,
            systemMemoryGB: Double(info.memorySize) / 1_073_741_824,
            gpuMemoryLimitGB: Double(info.maxRecommendedWorkingSetSize) / 1_073_741_824
        )
    }

    /// Format bytes as a human-readable string.
    static func formatGB(_ bytes: Int) -> String {
        String(format: "%.1fGB", Double(bytes) / 1_073_741_824)
    }

    /// Print hardware info as `[BENCH]` lines (matches mlx-swift-lm format
    /// so the same grep filter in the shell driver picks up both).
    static func printHardwareInfo() {
        let hw = hardware()
        print("[BENCH] Hardware: \(hw.architecture)")
        print("[BENCH] System RAM: \(String(format: "%.0f", hw.systemMemoryGB))GB")
        print("[BENCH] GPU Memory Limit: \(String(format: "%.0f", hw.gpuMemoryLimitGB))GB")
    }
}
