import Foundation
import MLX
import MLXNN
import HuggingFace
import MLXAudioCore

// MARK: - Fish S1 DAC Model

/// Fish S1 DAC audio codec used by Echo TTS.
/// A causal audio autoencoder with encoder, quantizer, and decoder.
public class FishS1DAC: Module {
    public let config: FishS1DACConfig
    public let sampleRate: Int
    public let hopLength: Int
    public let frameLength: Int

    @ModuleInfo(key: "encoder") var encoder: FishS1DACEncoder
    @ModuleInfo(key: "quantizer") var quantizer: FishDownsampleResidualVectorQuantize
    @ModuleInfo(key: "decoder") var decoder: FishS1DACDecoder

    public init(config: FishS1DACConfig) {
        self.config = config
        self.sampleRate = config.sampleRate
        self.hopLength = config.hopLength
        self.frameLength = config.frameLength

        self._encoder.wrappedValue = FishS1DACEncoder(config: config)
        self._quantizer.wrappedValue = FishDownsampleResidualVectorQuantize(config: config)
        self._decoder.wrappedValue = FishS1DACDecoder(config: config)
    }

    // MARK: - Padding

    private func padToFrameLength(_ audio: MLXArray) -> MLXArray {
        let length = audio.dim(-1)
        if length % frameLength != 0 {
            let padAmount = frameLength - (length % frameLength)
            return MLX.padded(audio, widths: [.init(0), .init(0), .init((0, padAmount))])
        }
        return audio
    }

    // MARK: - Encode

    /// Encode audio waveform to discrete indices.
    /// Input: [B, 1, T] (NCL). Returns (indices [B, nCodebooks+1, T_ds], indicesLens).
    public func encode(_ audio: MLXArray) -> (MLXArray, MLXArray) {
        let padded = padToFrameLength(audio)
        let z = encoder(padded)  // [B, latentDim, T_enc]
        let (_, indices) = quantizer(z)
        return (indices, MLXArray(indices.dim(-1)))
    }

    // MARK: - Decode

    /// Decode discrete indices to audio waveform.
    /// Input: indices [B, nCodebooks+1, T_ds]. Returns [B, 1, T].
    public func decode(_ indices: MLXArray) -> MLXArray {
        let zQ = quantizer.decodeFromIndices(indices)
        let zQProcessed = quantizer.postProcessAndUpsample(zQ)
        return decoder(zQProcessed)
    }

    // MARK: - encode_zq / decode_zq (Used by Echo TTS)

    /// Encode audio to quantized latent z_q (pre-upsample).
    /// This is the format Echo TTS works with.
    /// Input: [B, 1, T]. Returns [B, latentDim, T_ds] where T_ds = T / frameLength.
    public func encodeZQ(_ audio: MLXArray) -> MLXArray {
        let (indices, _) = encode(audio)

        // Clip indices to valid codebook ranges
        var clippedIndices = MLXArray.zeros(like: indices)

        // Semantic codebook (index 0): clip to semantic_codebook_size
        let semIdx = MLX.clip(
            indices[0..., ..<1, 0...],
            min: 0,
            max: config.semanticCodebookSize - 1
        )

        // Residual codebooks (index 1+): clip to codebook_size
        if indices.dim(1) > 1 {
            let resIdx = MLX.clip(
                indices[0..., 1..., 0...],
                min: 0,
                max: config.codebookSize - 1
            )
            clippedIndices = MLX.concatenated([semIdx, resIdx], axis: 1)
        } else {
            clippedIndices = semIdx
        }

        // Decode indices to continuous latent (pre-upsample)
        return quantizer.decodeFromIndices(clippedIndices)
    }

    /// Decode quantized latent z_q to audio waveform.
    /// Input: [B, latentDim, T_ds]. Returns [B, 1, T].
    public func decodeZQ(_ zQ: MLXArray) -> MLXArray {
        let zQProcessed = quantizer.postProcessAndUpsample(zQ)
        return decoder(zQProcessed)
    }

    // MARK: - Weight Sanitization

    /// Sanitize weights from PyTorch format.
    /// - Converts weight normalization parametrization names
    /// - Remaps decoder.model.{N} numbered keys to named keys (conv_in, blocks, snake_out, conv_out)
    /// - Remaps encoder.block.{N} numbered keys to named keys (conv_in, blocks, snake_out, conv_out)
    /// - Skips non-parameter keys (causal_mask, freqs_cis)
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip non-parameter keys
            if key.contains("causal_mask") || key.contains("freqs_cis") {
                continue
            }

            var newKey = key

            // 1) PyTorch weight normalization parametrization remapping
            if newKey.contains(".parametrizations.weight.original0") {
                newKey = newKey.replacingOccurrences(
                    of: ".parametrizations.weight.original0",
                    with: ".weight_g"
                )
            } else if newKey.contains(".parametrizations.weight.original1") {
                newKey = newKey.replacingOccurrences(
                    of: ".parametrizations.weight.original1",
                    with: ".weight_v"
                )
            }

            // 2) Decoder key remapping: decoder.model.{N}.xxx -> decoder.{named}.xxx
            if newKey.hasPrefix("decoder.model.") {
                newKey = remapDecoderKey(newKey)
            }

            // 3) Encoder key remapping: encoder.block.{N}.xxx -> encoder.{named}.xxx
            if newKey.hasPrefix("encoder.block.") {
                newKey = remapEncoderKey(newKey)
            }

            // Quantizer keys already match Swift structure (no remapping needed)

            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - Decoder Key Remapping

    /// Remap decoder.model.{N}.xxx to decoder.{named}.xxx
    /// Python nn.Sequential indices:
    ///   0 = CausalWNConv1d (conv_in)
    ///   1-4 = DecoderBlock (blocks.0-3)
    ///   5 = Snake1d (snake_out)
    ///   6 = CausalWNConv1d (conv_out)
    private static func remapDecoderKey(_ key: String) -> String {
        let prefix = "decoder.model."
        let afterPrefix = String(key.dropFirst(prefix.count))

        guard let dotIdx = afterPrefix.firstIndex(of: ".") else { return key }
        guard let modelIdx = Int(afterPrefix[afterPrefix.startIndex..<dotIdx]) else { return key }
        let rest = String(afterPrefix[afterPrefix.index(after: dotIdx)...])

        switch modelIdx {
        case 0:
            return "decoder.conv_in.\(rest)"
        case 1, 2, 3, 4:
            let blockIdx = modelIdx - 1
            return remapDecoderBlockKey(blockIdx: blockIdx, rest: rest)
        case 5:
            return "decoder.snake_out.\(rest)"
        case 6:
            return "decoder.conv_out.\(rest)"
        default:
            return key
        }
    }

    /// Remap decoder block sub-keys.
    /// Python DecoderBlock.block is nn.Sequential:
    ///   0 = Snake1d (snake)
    ///   1 = CausalWNConvTranspose1d (conv)
    ///   2-4 = ResidualUnit (res_units.0-2)
    private static func remapDecoderBlockKey(blockIdx: Int, rest: String) -> String {
        let blockPrefix = "block."
        guard rest.hasPrefix(blockPrefix) else {
            return "decoder.blocks.\(blockIdx).\(rest)"
        }
        let afterBlock = String(rest.dropFirst(blockPrefix.count))

        guard let dotIdx = afterBlock.firstIndex(of: ".") else {
            // Leaf key like "block.0" — shouldn't happen but handle gracefully
            return "decoder.blocks.\(blockIdx).\(rest)"
        }
        guard let subIdx = Int(afterBlock[afterBlock.startIndex..<dotIdx]) else {
            return "decoder.blocks.\(blockIdx).\(rest)"
        }
        let rest2 = String(afterBlock[afterBlock.index(after: dotIdx)...])

        switch subIdx {
        case 0:
            return "decoder.blocks.\(blockIdx).snake.\(rest2)"
        case 1:
            return "decoder.blocks.\(blockIdx).conv.\(rest2)"
        case 2, 3, 4:
            let resIdx = subIdx - 2
            return "decoder.blocks.\(blockIdx).res_units.\(resIdx).\(rest2)"
        default:
            return "decoder.blocks.\(blockIdx).\(rest)"
        }
    }

    // MARK: - Encoder Key Remapping

    /// Remap encoder.block.{N}.xxx to encoder.{named}.xxx
    /// Python nn.Sequential indices:
    ///   0 = CausalWNConv1d (conv_in)
    ///   1-4 = EncoderBlock (blocks.0-3)
    ///   5 = Snake1d (snake_out)
    ///   6 = CausalWNConv1d (conv_out)
    private static func remapEncoderKey(_ key: String) -> String {
        let prefix = "encoder.block."
        let afterPrefix = String(key.dropFirst(prefix.count))

        guard let dotIdx = afterPrefix.firstIndex(of: ".") else { return key }
        guard let blockIdx = Int(afterPrefix[afterPrefix.startIndex..<dotIdx]) else { return key }
        let rest = String(afterPrefix[afterPrefix.index(after: dotIdx)...])

        switch blockIdx {
        case 0:
            return "encoder.conv_in.\(rest)"
        case 1, 2, 3, 4:
            let encBlockIdx = blockIdx - 1
            return remapEncoderBlockKey(blockIdx: encBlockIdx, rest: rest)
        case 5:
            return "encoder.snake_out.\(rest)"
        case 6:
            return "encoder.conv_out.\(rest)"
        default:
            return key
        }
    }

    /// Remap encoder block sub-keys.
    /// Python EncoderBlock.block is nn.Sequential:
    ///   0-2 = ResidualUnit (res_units.0-2)
    ///   3 = Snake1d (snake)
    ///   4 = CausalWNConv1d (conv)
    ///   5 = WindowLimitedTransformer (transformer) — only in block 3
    private static func remapEncoderBlockKey(blockIdx: Int, rest: String) -> String {
        let blockPrefix = "block."
        guard rest.hasPrefix(blockPrefix) else {
            return "encoder.blocks.\(blockIdx).\(rest)"
        }
        let afterBlock = String(rest.dropFirst(blockPrefix.count))

        guard let dotIdx = afterBlock.firstIndex(of: ".") else {
            return "encoder.blocks.\(blockIdx).\(rest)"
        }
        guard let subIdx = Int(afterBlock[afterBlock.startIndex..<dotIdx]) else {
            return "encoder.blocks.\(blockIdx).\(rest)"
        }
        let rest2 = String(afterBlock[afterBlock.index(after: dotIdx)...])

        switch subIdx {
        case 0, 1, 2:
            return "encoder.blocks.\(blockIdx).res_units.\(subIdx).\(rest2)"
        case 3:
            return "encoder.blocks.\(blockIdx).snake.\(rest2)"
        case 4:
            return "encoder.blocks.\(blockIdx).conv.\(rest2)"
        case 5:
            return "encoder.blocks.\(blockIdx).transformer.\(rest2)"
        default:
            return "encoder.blocks.\(blockIdx).\(rest)"
        }
    }

    // MARK: - Load from Pretrained

    public static func fromPretrained(
        _ repoId: String,
        cache: HubCache = .default
    ) async throws -> FishS1DAC {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NSError(
                domain: "FishS1DAC",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        var config = FishS1DACConfig()
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(FishS1DACConfig.self, from: configData)
        }

        // Create model
        let model = FishS1DAC(config: config)

        // Load weights
        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        var weights: [String: MLXArray]
        if FileManager.default.fileExists(atPath: weightsURL.path) {
            weights = try loadArrays(url: weightsURL)
        } else {
            let torchURL = modelDir.appendingPathComponent("pytorch_model.safetensors")
            weights = try loadArrays(url: torchURL)
        }

        let sanitizedWeights = sanitize(weights)

        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedWeights),
            verify: .noUnusedKeys
        )

        eval(model.parameters())
        return model
    }
}

// MARK: - AudioCodecModel Conformance

extension FishS1DAC: AudioCodecModel {
    public typealias EncodedAudio = MLXArray

    public var codecSampleRate: Double? { Double(sampleRate) }

    public func encodeAudio(_ waveform: MLXArray) -> MLXArray {
        encodeZQ(waveform)
    }

    public func decodeAudio(_ input: MLXArray) -> MLXArray {
        decodeZQ(input)
    }
}
