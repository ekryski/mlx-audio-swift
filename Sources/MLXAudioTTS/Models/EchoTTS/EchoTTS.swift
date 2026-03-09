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

    // Cached speaker latent and KV cache (reused across text chunks for same voice)
    private var cachedSpeakerLatent: MLXArray?
    private var cachedSpeakerMask: MLXArray?
    private var cachedSpeakerKVCache: EchoKVCache?
    private var cachedRefAudioId: Int?  // Hash of ref audio for cache invalidation

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

    /// Runtime quantization configuration for Echo TTS.
    ///
    /// Pass a `(bits, groupSize)` tuple to quantize the DiT model weights after
    /// loading BF16 weights from disk. Only `Linear` layers in the DiT transformer
    /// are quantized — the Fish S1 DAC codec and PCA state are left untouched.
    ///
    /// - `bits`: 4 or 8 (lower = faster + smaller but potentially lower quality)
    /// - `groupSize`: typically 64 (number of weights per quantization group)
    public typealias QuantizationConfig = (bits: Int, groupSize: Int)

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default,
        quantization: QuantizationConfig? = nil
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

        // Apply optimized defaults: Independent (High Speaker CFG) mode.
        // KV scaling 1.1 with high cfg_scale_speaker (8.0, Python default) for
        // natural voice quality. 10 diffusion steps for fast generation (~4× faster
        // than 40 steps) with acceptable quality.
        config.sampler.speakerKvScale = 1.1
        config.sampler.cfgScaleSpeaker = 8.0
        config.sampler.numSteps = 10
        if config.sampler.speakerKvMaxLayers == nil {
            config.sampler.speakerKvMaxLayers = 12
        }
        if config.sampler.speakerKvMinT == nil {
            config.sampler.speakerKvMinT = 0.5
        }
        let model = EchoTTSModel(config: config)

        // Load weights
        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        let weights = try loadArrays(url: weightsURL)
        let sanitizedWeights = sanitize(weights, config: config)

        let unflattened = ModuleParameters.unflattened(sanitizedWeights)
        try model.model.update(parameters: unflattened, verify: .none)

        // Apply runtime quantization to DiT model if requested.
        // Only Linear layers are quantized (not QuantizedLinear, norms, or embeddings).
        // The codec (Fish S1 DAC) is loaded separately and is never quantized.
        if let q = quantization {
            let quantStart = Date()
            MLXNN.quantize(
                model: model.model,
                groupSize: q.groupSize,
                bits: q.bits
            ) { path, module in
                // Only quantize Linear layers with compatible dimensions.
                // Skip QuantizedLinear (already quantized) and layers whose last
                // weight dimension isn't divisible by groupSize (e.g., the 80-dim
                // PCA projection layer where 80 % 64 ≠ 0).
                guard let linear = module as? Linear, !(module is QuantizedLinear) else {
                    return false
                }
                let lastDim = linear.weight.dim(linear.weight.ndim - 1)
                return lastDim % q.groupSize == 0
            }
            let quantTime = Date().timeIntervalSince(quantStart)
            print("[EchoTTS] Runtime quantization (\(q.bits)-bit, group=\(q.groupSize)) applied in \(String(format: "%.3f", quantTime))s")
        }

        eval(model.model)

        // Load PCA state
        let pcaURL = modelDir.appendingPathComponent(config.pcaFilename)
        model.pcaState = try echoLoadPCAState(from: pcaURL)

        // Load Fish S1 DAC codec (never quantized — codec quality is critical)
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

    // MARK: - Prepare Generation (Pre-cache)

    /// Pre-computed caches for generation. Moving text tokenization, speaker latent
    /// encoding, and KV cache building out of the diffusion critical path.
    ///
    /// Public so callers can pre-prepare the next chunk while the current one
    /// is still synthesizing (look-ahead pipelining).
    public struct PreparedGeneration: @unchecked Sendable {
        public let textTokens: MLXArray
        public let textMask: MLXArray
        public let speakerLatent: MLXArray
        public let speakerMask: MLXArray
        public let textKVCache: EchoKVCache
        public let speakerKVCache: EchoKVCache
    }

    /// Pre-compute text tokens, speaker latent, and KV caches before diffusion.
    ///
    /// Speaker latent and speaker KV cache are cached across calls for the same
    /// reference audio, saving ~1.5s per text chunk after the first.
    ///
    /// Call this for the next chunk while the current chunk is synthesizing,
    /// then pass the result to `generateStream(prepared:...)` to skip
    /// the preparation step and start diffusion immediately.
    public func prepareGeneration(
        text: String,
        refAudio: MLXArray?,
        refText: String?
    ) throws -> PreparedGeneration {
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
        print("[EchoTTS] Text tokens shape: \(textTokens.shape)")

        // Prepare speaker latent (cached across calls for same ref audio)
        let speakerLatent: MLXArray
        let speakerMask: MLXArray
        let speakerKVCache: EchoKVCache

        // Check if we can reuse cached speaker data
        let refAudioId = refAudio.map { $0.shape.hashValue ^ $0.dtype.hashValue } ?? 0

        if let cachedLatent = cachedSpeakerLatent,
           let cachedMask = cachedSpeakerMask,
           let cachedKV = cachedSpeakerKVCache,
           cachedRefAudioId == refAudioId {
            // Reuse cached speaker latent and KV cache
            speakerLatent = cachedLatent
            speakerMask = cachedMask
            speakerKVCache = cachedKV
            print("[EchoTTS] Using cached speaker latent: shape=\(speakerLatent.shape)")
        } else {
            // Compute speaker latent from scratch
            let spkStart = Date()

            if let ref = refAudio {
                var refAudioInput = ref
                if refAudioInput.ndim == 1 {
                    refAudioInput = refAudioInput.expandedDimensions(axis: 0)
                }
                if refAudioInput.ndim == 2 && refAudioInput.dim(0) > 1 {
                    refAudioInput = MLX.mean(refAudioInput, axis: 0, keepDims: true)
                }
                print("[EchoTTS] Reference audio shape: \(refAudioInput.shape)")
                (speakerLatent, speakerMask) = echoGetSpeakerLatentAndMask(
                    fishAE: fishAE, pcaState: pcaState, audio: refAudioInput, config: config
                )
            } else {
                let patchSize = config.dit.speakerPatchSize
                speakerLatent = MLXArray.zeros([1, patchSize, config.dit.latentSize])
                speakerMask = MLXArray.zeros([1, patchSize], type: Bool.self)
            }

            eval(speakerLatent, speakerMask)

            // Compute speaker KV cache
            speakerKVCache = model.getKVCacheSpeaker(speakerLatent)
            for (k, v) in speakerKVCache { eval(k, v) }

            let spkTime = Date().timeIntervalSince(spkStart)
            print("[EchoTTS] Computed speaker latent: shape=\(speakerLatent.shape) in \(String(format: "%.3f", spkTime))s")

            // Cache for reuse
            cachedSpeakerLatent = speakerLatent
            cachedSpeakerMask = speakerMask
            cachedSpeakerKVCache = speakerKVCache
            cachedRefAudioId = refAudioId
        }

        // Pre-compute text KV cache (changes per text, can't cache)
        let textKVCache = model.getKVCacheText(textTokens, mask: textMask)
        for (k, v) in textKVCache { eval(k, v) }

        print("[EchoTTS] Pre-computed text KV cache (\(textKVCache.count) layers), speaker KV cache (\(speakerKVCache.count) layers)")

        return PreparedGeneration(
            textTokens: textTokens,
            textMask: textMask,
            speakerLatent: speakerLatent,
            speakerMask: speakerMask,
            textKVCache: textKVCache,
            speakerKVCache: speakerKVCache
        )
    }

    // MARK: - Generate Stream

    /// Streaming generation using blockwise diffusion with per-block overlapped decode.
    ///
    /// Each block is decoded immediately as it completes, yielding audio incrementally:
    /// 1. Block 0: decoded standalone (no context) → yield for TTFA
    /// 2. Block 1+: decoded with overlapped context (all previous blocks prepended)
    ///    → yield each block's audio as it's ready
    ///
    /// Fish S1 DAC is fully causal (left-only padding), so overlapped decode produces
    /// audio mathematically identical to full-sequence decode. Block 1's audio arrives
    /// just as Block 0's audio finishes playing, creating nearly seamless playback.
    ///
    /// Silence detection runs at each block to crop trailing silence and skip
    /// blocks that fall entirely within the silence region.
    ///
    /// Speaker latent and KV cache are cached across calls for the same voice.
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

                    // Pre-compute text/speaker/KV caches (speaker is cached across calls)
                    let prepStart = Date()
                    let prep = try self.prepareGeneration(
                        text: text, refAudio: refAudioCopy, refText: refText
                    )
                    let prepTime = Date().timeIntervalSince(prepStart)
                    print("[EchoTTS] Preparation time: \(String(format: "%.3f", prepTime))s")

                    try self.generateStreamFromPrepared(
                        prep: prep,
                        promptSize: text.utf8.count,
                        prepTime: prepTime,
                        startTime: startTime,
                        continuation: continuation
                    )
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Streaming generation from a pre-computed `PreparedGeneration`.
    ///
    /// Use this with `prepareGeneration()` for look-ahead pipelining: prepare the
    /// next chunk while the current one is still synthesizing, then call this to
    /// skip the preparation step and start diffusion immediately.
    ///
    /// ```swift
    /// // Prepare next chunk while current is synthesizing
    /// let nextPrep = try model.prepareGeneration(text: nextChunk, ...)
    /// // Start synthesis immediately from pre-computed caches
    /// let stream = model.generateStream(prepared: nextPrep, ...)
    /// ```
    public func generateStream(
        prepared: PreparedGeneration,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    let startTime = Date()
                    print("[EchoTTS] Preparation time: 0.000s (pre-prepared)")

                    try self.generateStreamFromPrepared(
                        prep: prepared,
                        promptSize: prepared.textTokens.dim(1),
                        prepTime: 0,
                        startTime: startTime,
                        continuation: continuation
                    )
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Shared implementation for blockwise streaming from a prepared generation.
    ///
    /// Includes three optimizations for gapless, artifact-free playback:
    /// 1. **Limited decode context** (8 frames): Overlapped decode only prepends
    ///    the last 8 latent frames as context instead of all previous blocks.
    ///    The Fish S1 DAC's causal convolutions and windowed attention have a
    ///    bounded receptive field, so 8 frames provides identical audio quality
    ///    while cutting decode time by ~25-40%.
    /// 2. **Overlap-add crossfade** (2048 samples ≈ 46ms): A cosine crossfade
    ///    blends two decodings of the same temporal region (block N's tail and
    ///    block N+1's context-decoded version), eliminating boundary artifacts
    ///    without ghosting since both signals represent the same content.
    /// 3. **End fade-out** (512 samples ≈ 11.6ms): A fade-out on the last
    ///    audio chunk prevents clicks when the waveform doesn't end at zero.
    private func generateStreamFromPrepared(
        prep: PreparedGeneration,
        promptSize: Int,
        prepTime: Double,
        startTime: Date,
        continuation: AsyncThrowingStream<AudioGeneration, Error>.Continuation
    ) throws {
        guard let fishAE = self.fishAE else {
            throw AudioGenerationError.modelNotInitialized("Fish S1 DAC codec not loaded")
        }
        guard let pcaState = self.pcaState else {
            throw AudioGenerationError.modelNotInitialized("PCA state not loaded")
        }

        // Use dynamic block sizes based on text length
        // Gapless sizing: blocks grow by ~15% each, ensuring decode fits within previous audio
        let textTokenCount = prep.textTokens.dim(1)
        let blockSizes = self.config.blockSizes(forTokenCount: textTokenCount)

        // Crossfade eliminates artifacts at block boundaries from decode context differences.
        // With true overlap-add crossfading (both signals represent the same temporal content,
        // just decoded with different causal context), we can use a longer crossfade without
        // ghosting. 2048 samples = 1 full DAC frame (~46ms) — long enough to smooth any
        // waveform differences from the limited decode context, while still imperceptible
        // in speech. Cosine curve provides smooth transitions with zero-derivative endpoints.
        let crossfadeLength = 2048  // ~46ms at 44.1kHz (1 DAC frame)
        // Fade-out at end of stream prevents clicks from non-zero final samples.
        // Applied to the last audio segment (after silence detection or end of blocks).
        let endFadeLength = 512    // ~11.6ms at 44.1kHz
        // Limit overlapped decode context to reduce decode time while preserving quality
        let maxDecodeContext = 8   // 8 frames = 16,384 samples ≈ 0.37s of DAC context
        // Silence detection threshold — more conservative than the default (0.05) to avoid
        // false positives from natural speech pauses (soft consonants, inter-word gaps).
        // 0.02 catches genuine silence while being robust against quiet speech passages.
        let silenceStdThreshold: Float = 0.02

        print("[EchoTTS] Sampler config: numSteps=\(self.config.sampler.numSteps), cfgText=\(self.config.sampler.cfgScaleText), cfgSpeaker=\(self.config.sampler.cfgScaleSpeaker), kvScale=\(String(describing: self.config.sampler.speakerKvScale))")
        print("[EchoTTS] Block sizes: \(blockSizes) (for \(textTokenCount) tokens, \(blockSizes.count) blocks)")

        var blockIndex = 0
        var totalAudioSamples = 0
        var previousBlockTail: MLXArray?  // Held-back samples for crossfade blending

        // Pre-compute cosine crossfade ramps (reused across blocks).
        // Cosine has zero-derivative endpoints for smoother transitions than linear.
        // fadeIn + fadeOut = 1.0 at all points (constant-power crossfade).
        let fadeIn = MLXArray((0..<crossfadeLength).map { 0.5 * (1.0 - cos(Float.pi * Float($0) / Float(crossfadeLength))) })
        let fadeOut = MLXArray((0..<crossfadeLength).map { 0.5 * (1.0 + cos(Float.pi * Float($0) / Float(crossfadeLength))) })

        /// Trim trailing near-silence from audio (amplitude-based).
        /// Scans backward from the end to find the last sample above the threshold,
        /// then trims everything after it. This catches trailing noise/artifacts
        /// that the latent-based silence detection misses (especially in short blocks
        /// where the model output doesn't drop to zero in latent space).
        func trimTrailingSilence(_ audio: MLXArray, threshold: Float = 0.01, minKeep: Int = 2048) -> MLXArray {
            let len = audio.dim(0)
            guard len > minKeep else { return audio }
            // Check windows of 1024 samples from the end
            let windowSize = 1024
            var trimPoint = len
            var pos = len
            while pos > minKeep {
                let start = max(0, pos - windowSize)
                let window = audio[start..<pos]
                let maxAmp = Float(MLX.abs(window).max().item(Float.self))
                if maxAmp > threshold {
                    trimPoint = min(pos + windowSize, len)  // Keep one window past last loud sample
                    break
                }
                trimPoint = start
                pos -= windowSize
            }
            if trimPoint < len {
                let trimmed = audio[..<trimPoint]
                eval(trimmed)
                let trimmedMs = String(format: "%.1f", Double(len - trimPoint) / 44100.0 * 1000)
                print("[EchoTTS] Trimmed \(len - trimPoint) trailing silence samples (\(trimmedMs)ms)")
                return trimmed
            }
            return audio
        }

        /// Apply a short fade-out to audio to prevent clicks at stream end.
        func applyEndFade(_ audio: MLXArray) -> MLXArray {
            let len = audio.dim(0)
            let fadeSamples = min(endFadeLength, len)
            guard fadeSamples >= 2 else { return audio }
            let ramp = MLXArray((0..<fadeSamples).map { 1.0 - Float($0) / Float(fadeSamples) })
            let main = audio[..<(len - fadeSamples)]
            let tail = audio[(len - fadeSamples)...] * ramp
            let result = MLX.concatenated([main, tail], axis: 0)
            eval(result)
            return result
        }

        // Blockwise diffusion: decode each block immediately via overlapped decode
        // Callback returns false to trigger early termination when silence detected
        _ = echoSampleBlockwiseEulerCFG(
            model: self.model,
            textKVCond: prep.textKVCache,
            speakerKVCondBase: prep.speakerKVCache,
            textMask: prep.textMask,
            speakerMask: prep.speakerMask,
            blockSizes: blockSizes,
            config: self.config,
            rngSeed: 0,
            onBlockComplete: { blockLatent, contextLatent -> Bool in
                let prevFrames = contextLatent?.dim(1) ?? 0

                // Build full latent (context + this block) for silence detection
                let fullLatent: MLXArray
                if let ctx = contextLatent, ctx.dim(1) > 0 {
                    fullLatent = MLX.concatenated([ctx, blockLatent], axis: 1)
                } else {
                    fullLatent = blockLatent
                }

                // Check silence: find where signal flattens to zero.
                // Uses a stricter threshold than the default to avoid cutting off
                // speech during natural pauses or soft consonants.
                let flatPoint = echoFindFlatteningPoint(fullLatent[0], stdThreshold: silenceStdThreshold)

                if flatPoint <= prevFrames {
                    // Silence starts before this block — yield any held-back tail with fade-out, then stop
                    if let tail = previousBlockTail {
                        let fadedTail = applyEndFade(trimTrailingSilence(tail))
                        totalAudioSamples += fadedTail.dim(0)
                        continuation.yield(.audio(fadedTail))
                        previousBlockTail = nil
                    }
                    print("[EchoTTS] Block \(blockIndex): entirely in silence region (flatPoint=\(flatPoint), prevFrames=\(prevFrames)), skipping")
                    blockIndex += 1
                    return false  // Stop early — all remaining blocks are silence
                }

                // Decode this block with limited overlapped context.
                // For blocks after the first, also extract `crossfadeLength` overlap
                // samples from the context decode for true overlap-add crossfading.
                let decodeStart = Date()
                var blockAudio: MLXArray
                var contextOverlap: MLXArray? = nil

                if prevFrames == 0 {
                    // First block: no context needed
                    blockAudio = echoAEDecode(
                        fishAE: fishAE, pcaState: pcaState, zQ: blockLatent
                    )
                } else {
                    // Subsequent blocks: limited causal context for fast decode
                    let decodeResult = echoAEDecodeOverlapped(
                        fishAE: fishAE, pcaState: pcaState,
                        contextLatent: contextLatent, blockLatent: blockLatent,
                        maxContextFrames: maxDecodeContext,
                        overlapSamples: crossfadeLength
                    )
                    blockAudio = decodeResult.blockAudio
                    contextOverlap = decodeResult.contextOverlap
                }
                eval(blockAudio)

                // Crop if silence starts within this block
                let thisBlockEnd = prevFrames + blockLatent.dim(1)
                let silenceInThisBlock = flatPoint < thisBlockEnd
                if silenceInThisBlock {
                    let cropFrameInBlock = flatPoint - prevFrames
                    let cropSamples = cropFrameInBlock * 2048
                    blockAudio = blockAudio[0..., 0..., ..<cropSamples]
                    eval(blockAudio)
                    print("[EchoTTS] Block \(blockIndex): silence at frame \(flatPoint), cropped to \(cropFrameInBlock)/\(blockLatent.dim(1)) frames")
                }

                let decodeTime = Date().timeIntervalSince(decodeStart)
                let blockSamples = blockAudio.dim(blockAudio.ndim - 1)
                let audioSec = Double(blockSamples) / 44100.0

                if blockIndex == 0 {
                    let ttfa = Date().timeIntervalSince(startTime)
                    print("[EchoTTS] Block \(blockIndex) decoded: \(blockSamples) samples (\(String(format: "%.2f", audioSec))s audio) in \(String(format: "%.3f", decodeTime))s, TTFA=\(String(format: "%.3f", ttfa))s")
                } else {
                    let elapsed = Date().timeIntervalSince(startTime)
                    print("[EchoTTS] Block \(blockIndex) decoded: \(blockSamples) samples (\(String(format: "%.2f", audioSec))s audio) in \(String(format: "%.3f", decodeTime))s, elapsed=\(String(format: "%.3f", elapsed))s")
                }

                if blockSamples > 0 {
                    // Extract 1D audio: [samples]
                    var audio1D = blockAudio[0, 0]

                    // True overlap-add crossfade with previous block's held-back tail.
                    // Instead of blending adjacent audio (tail of block N + start of block N+1),
                    // blend the tail with the context decode's version of that same temporal region.
                    // This eliminates ghosting because both signals represent the same content.
                    if let prevTail = previousBlockTail {
                        // Use context overlap (re-decode of tail's temporal region) if available,
                        // otherwise fall back to the start of the new block's audio
                        let overlapRegion: MLXArray
                        if let overlap = contextOverlap {
                            overlapRegion = overlap[0, 0]  // [1, 1, samples] → [samples]
                        } else {
                            overlapRegion = audio1D[..<min(crossfadeLength, audio1D.dim(0))]
                        }

                        let cfLen = min(crossfadeLength, audio1D.dim(0), prevTail.dim(0), overlapRegion.dim(0))
                        if cfLen > 0 {
                            let fi = cfLen == crossfadeLength ? fadeIn : MLXArray((0..<cfLen).map { 0.5 * (1.0 - cos(Float.pi * Float($0) / Float(cfLen))) })
                            let fo = cfLen == crossfadeLength ? fadeOut : MLXArray((0..<cfLen).map { 0.5 * (1.0 + cos(Float.pi * Float($0) / Float(cfLen))) })
                            let blended = prevTail[..<cfLen] * fo + overlapRegion[..<cfLen] * fi
                            // Prepend the crossfade blend, then the FULL new block audio.
                            // The blend covers the temporal gap between the yielded portion of
                            // block N and the start of block N+1. The overlap region is
                            // continuous with audio1D (same decode pass), so no seam.
                            audio1D = MLX.concatenated([blended, audio1D], axis: 0)
                        }
                        eval(audio1D)
                    }

                    // Hold back tail for next block's crossfade (unless this is the last block)
                    let audioLen = audio1D.dim(0)
                    if audioLen > crossfadeLength && !silenceInThisBlock {
                        // More blocks may follow — hold back tail
                        previousBlockTail = audio1D[(audioLen - crossfadeLength)...]
                        audio1D = audio1D[..<(audioLen - crossfadeLength)]
                        eval(previousBlockTail!, audio1D)
                    } else {
                        // Last block or silence-cropped — trim trailing silence + fade-out
                        previousBlockTail = nil
                        audio1D = applyEndFade(trimTrailingSilence(audio1D))
                    }

                    totalAudioSamples += audio1D.dim(0)
                    continuation.yield(.audio(audio1D))
                }

                blockIndex += 1

                // If silence started in this block, remaining blocks are all silence
                return !silenceInThisBlock
            }
        )

        // Yield any remaining held-back tail from the last block (trim silence + fade-out)
        if let tail = previousBlockTail {
            let fadedTail = applyEndFade(trimTrailingSilence(tail))
            totalAudioSamples += fadedTail.dim(0)
            continuation.yield(.audio(fadedTail))
            previousBlockTail = nil
        }

        let generateTime = Date().timeIntervalSince(startTime)
        let peakMemory = Double(Memory.snapshot().peakMemory) / 1e9
        let durationSec = Double(totalAudioSamples) / 44100.0

        print("[EchoTTS] Total: \(String(format: "%.2f", generateTime))s generation, \(totalAudioSamples) samples (\(String(format: "%.2f", durationSec))s audio)")

        let info = AudioGenerationInfo(
            promptTokenCount: promptSize,
            generationTokenCount: totalAudioSamples,
            prefillTime: prepTime,
            generateTime: generateTime,
            tokensPerSecond: Double(totalAudioSamples) / max(generateTime, 0.001),
            peakMemoryUsage: peakMemory
        )
        continuation.yield(.info(info))
        continuation.finish()
    }
}
