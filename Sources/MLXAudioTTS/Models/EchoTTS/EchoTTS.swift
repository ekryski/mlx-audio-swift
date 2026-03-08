import Foundation
import MLX
import MLXNN
import HuggingFace
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon

// MARK: - Echo TTS Model

public final class EchoTTSModel: SpeechGenerationModel, @unchecked Sendable {
    public var config: EchoTTSConfig
    let model: EchoDiT
    var fishAE: FishS1DAC?
    var pcaState: EchoPCAState?

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: config.sampler.sequenceLength,
            temperature: 1.0,
            topP: 1.0,
            repetitionPenalty: 1.0,
            repetitionContextSize: 0
        )
    }

    init(config: EchoTTSConfig) {
        self.config = config
        self.model = EchoDiT(config.dit, hasLatentAttention: !config.deleteBlockwiseModules)
    }

    // MARK: - Weight Sanitization

    static func sanitize(_ weights: [String: MLXArray], config: EchoTTSConfig) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        let pcaKeys = Set(["pca_components", "pca_mean", "latent_scale"])
        let blockwiseKeys = ["wk_latent", "wv_latent", "latent_encoder", "latent_norm"]

        for (key, value) in weights {
            // Skip PCA state keys
            if pcaKeys.contains(key) { continue }

            var newKey = key

            // Strip "model." prefix since we load directly into the EchoDiT instance
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst(6))
            }

            // Skip blockwise module keys if configured
            if config.deleteBlockwiseModules {
                if blockwiseKeys.contains(where: { newKey.contains($0) }) { continue }
            }

            // Remap Sequential naming: cond_module.0.weight -> cond_module.layers.0.weight
            if newKey.contains("cond_module.") && !newKey.contains("cond_module.layers.") {
                newKey = newKey.replacingOccurrences(
                    of: "cond_module.",
                    with: "cond_module.layers."
                )
            }

            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - Load from Pretrained

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> EchoTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )

        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        var config = try JSONDecoder().decode(EchoTTSConfig.self, from: configData)

        // Apply optimized defaults
        // 30 steps produces nearly identical quality to 40, ~25% faster
        if config.sampler.numSteps > 30 {
            config.sampler.numSteps = 30
        }
        // config.json ships cfg_scale_speaker=8.0, but 5.0 produces cleaner output
        if config.sampler.cfgScaleSpeaker > 5.0 {
            config.sampler.cfgScaleSpeaker = 5.0
        }
        // NOTE: speaker_kv_scale is left as nil (disabled) intentionally.
        // Enabling it (e.g. 1.5) combined with cfg_scale_speaker >= 5.0 causes the
        // model to over-condition on the speaker, producing extremely long outputs
        // (21s instead of 4s) that sound like slow motion. If enabling KV scaling,
        // cfg_scale_speaker must be reduced to ~3.0 to compensate.
        let model = EchoTTSModel(config: config)

        // Load weights
        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        let weights = try loadArrays(url: weightsURL)
        let sanitizedWeights = sanitize(weights, config: config)

        let unflattened = ModuleParameters.unflattened(sanitizedWeights)
        try model.model.update(parameters: unflattened, verify: .none)
        eval(model.model)

        // Load PCA state
        let pcaURL = modelDir.appendingPathComponent(config.pcaFilename)
        model.pcaState = try echoLoadPCAState(from: pcaURL)

        // Load Fish S1 DAC codec
        model.fishAE = try await FishS1DAC.fromPretrained(config.fishCodecRepo, cache: cache)

        return model
    }

    // MARK: - Generate

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard let fishAE = fishAE else {
            throw AudioGenerationError.modelNotInitialized("Fish S1 DAC codec not loaded")
        }
        guard let pcaState = pcaState else {
            throw AudioGenerationError.modelNotInitialized("PCA state not loaded")
        }

        // Prepare text
        let (textTokens, textMask, normalizedTexts) = echoGetTextInputIdsAndMask(
            [text], maxLength: config.maxTextLength, normalize: config.normalizeText
        )
        eval(textTokens, textMask)
        print("[EchoTTS] Normalized text: \(normalizedTexts)")
        print("[EchoTTS] Text tokens shape: \(textTokens.shape), first 20: \(textTokens[0, ..<min(20, textTokens.dim(1))].asArray(Int32.self))")
        print("[EchoTTS] Sampler config: numSteps=\(config.sampler.numSteps), cfgText=\(config.sampler.cfgScaleText), cfgSpeaker=\(config.sampler.cfgScaleSpeaker), kvScale=\(String(describing: config.sampler.speakerKvScale))")

        // Prepare speaker latent
        let speakerLatent: MLXArray
        let speakerMask: MLXArray

        if let ref = refAudio {
            // ref: expected shape [samples] or [1, samples] or [1, 1, samples]
            var refAudioInput = ref
            if refAudioInput.ndim == 1 {
                // [samples] -> [1, samples] (Python: audio = audio[None, :])
                refAudioInput = refAudioInput.expandedDimensions(axis: 0)
            }
            if refAudioInput.ndim == 2 && refAudioInput.dim(0) > 1 {
                // Multi-channel: average to mono
                refAudioInput = MLX.mean(refAudioInput, axis: 0, keepDims: true)
            }
            // Now: [1, samples]
            print("[EchoTTS] Reference audio shape: \(refAudioInput.shape)")
            (speakerLatent, speakerMask) = echoGetSpeakerLatentAndMask(
                fishAE: fishAE, pcaState: pcaState, audio: refAudioInput, config: config
            )
        } else {
            // Default: zero speaker latent (Python: mx.zeros((1, speaker_patch_size, latent_size)))
            let patchSize = config.dit.speakerPatchSize
            speakerLatent = MLXArray.zeros([1, patchSize, config.dit.latentSize])
            // Python: speaker_mask = mx.zeros((1, speaker_latent.shape[1]), dtype=mx.bool_)
            speakerMask = MLXArray.zeros([1, patchSize], type: Bool.self)
        }

        eval(speakerLatent, speakerMask)
        print("[EchoTTS] Speaker latent: shape=\(speakerLatent.shape), min=\(speakerLatent.min().item(Float.self)), max=\(speakerLatent.max().item(Float.self)), mean=\(MLX.mean(speakerLatent).item(Float.self))")
        print("[EchoTTS] Speaker mask: shape=\(speakerMask.shape), true_count=\(MLX.sum(speakerMask.asType(.int32)).item(Int32.self))")

        // Generate latents via diffusion sampling
        let latents = echoSampleEulerCFG(
            model: model,
            textTokens: textTokens,
            textMask: textMask,
            speakerLatent: speakerLatent,
            speakerMask: speakerMask,
            config: config,
            rngSeed: 0
        )

        eval(latents)
        print("[EchoTTS] Latent output: shape=\(latents.shape), min=\(latents.min().item(Float.self)), max=\(latents.max().item(Float.self))")

        // Check flattening point
        let flatPoint = echoFindFlatteningPoint(latents[0])
        print("[EchoTTS] Flattening point: \(flatPoint) frames = \(Double(flatPoint) * 2048.0 / 44100.0)s")

        // Decode latents to audio
        let audioOut = echoAEDecode(fishAE: fishAE, pcaState: pcaState, zQ: latents)

        eval(audioOut)
        print("[EchoTTS] Raw audio: shape=\(audioOut.shape)")

        // Crop at flattening point (Python: crop_audio_to_flattening_point(audio_out, latent_out[0]))
        let croppedAudio = echoCropAudioToFlatteningPoint(
            audio: audioOut, latent: latents[0]  // latents[0] is [T, latentSize]
        )

        eval(croppedAudio)
        let finalAudio = croppedAudio[0, 0]
        let durationSec = Double(finalAudio.dim(0)) / 44100.0
        print("[EchoTTS] Final audio: \(finalAudio.dim(0)) samples = \(durationSec)s")

        // Return as 1D audio: [samples] (Python: audio = audio_out[0, 0])
        return finalAudio
    }

    // MARK: - Generate Stream

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        nonisolated(unsafe) let refAudioCopy = refAudio
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    let startTime = Date()

                    let audio = try await self.generate(
                        text: text,
                        voice: voice,
                        refAudio: refAudioCopy,
                        refText: refText,
                        language: language,
                        generationParameters: generationParameters
                    )

                    let generateTime = Date().timeIntervalSince(startTime)
                    let peakMemory = Double(Memory.snapshot().peakMemory) / 1e9

                    let info = AudioGenerationInfo(
                        promptTokenCount: text.utf8.count,
                        generationTokenCount: audio.dim(0),
                        prefillTime: 0,
                        generateTime: generateTime,
                        tokensPerSecond: Double(audio.dim(0)) / max(generateTime, 0.001),
                        peakMemoryUsage: peakMemory
                    )

                    continuation.yield(.info(info))
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
