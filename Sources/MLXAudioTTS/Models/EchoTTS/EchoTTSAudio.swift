import Foundation
import MLX
import MLXAudioCodecs

// MARK: - PCA State

struct EchoPCAState {
    let pcaComponents: MLXArray  // [inputDim, latentSize]
    let pcaMean: MLXArray        // [inputDim]
    let latentScale: Float
}

func echoLoadPCAState(from url: URL) throws -> EchoPCAState {
    let weights = try loadArrays(url: url)
    guard let components = weights["pca_components"],
          let mean = weights["pca_mean"] else {
        throw NSError(
            domain: "EchoTTS",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "PCA state missing pca_components or pca_mean"]
        )
    }

    let scale: Float
    if let scaleArray = weights["latent_scale"] {
        scale = scaleArray.item(Float.self)
    } else {
        scale = 1.0
    }

    return EchoPCAState(
        pcaComponents: components,
        pcaMean: mean,
        latentScale: scale
    )
}

// MARK: - Audio Encoding/Decoding with PCA

/// Encode audio through Fish S1 DAC + PCA to 80-dim latent.
/// Input: [B, 1, T]. Returns [B, T_ds, latentSize].
func echoAEEncode(fishAE: FishS1DAC, pcaState: EchoPCAState, audio: MLXArray) -> MLXArray {
    // Fish encode_zq: [B, 1, T] -> [B, 1024, T_ds]
    // Python: fish_ae.encode_zq(audio).astype(mx.float32)
    let zQ = fishAE.encodeZQ(audio).asType(.float32)  // NCL

    // NCL -> NLC
    var z = zQ.transposed(0, 2, 1)  // [B, T_ds, 1024]

    // PCA transform: (z - mean) @ components.T * scale
    // components is [80, 1024]; we need [1024, 80] for the matmul
    z = MLX.matmul(z - pcaState.pcaMean, pcaState.pcaComponents.transposed(1, 0))
    z = z * pcaState.latentScale

    return z  // [B, T_ds, latentSize=80]
}

/// Decode 80-dim latent through inverse PCA + Fish S1 DAC.
/// Input: [B, T_ds, latentSize]. Returns [B, 1, T].
func echoAEDecode(fishAE: FishS1DAC, pcaState: EchoPCAState, zQ: MLXArray) -> MLXArray {
    // Inverse PCA: (z / scale) @ components + mean
    var z = zQ / pcaState.latentScale
    z = MLX.matmul(z, pcaState.pcaComponents) + pcaState.pcaMean  // [B, T_ds, 1024]

    // NLC -> NCL
    z = z.transposed(0, 2, 1)  // [B, 1024, T_ds]

    // Fish decode_zq: [B, 1024, T_ds] -> [B, 1, T]
    // Python: fish_ae.decode_zq(z_q.astype(mx.float32)).astype(mx.float32)
    return fishAE.decodeZQ(z.asType(.float32)).asType(.float32)
}

// MARK: - Overlapped Causal Decode

/// Result from overlapped decode, containing the block's audio and optional
/// overlap samples from the context decode for crossfading.
struct EchoOverlappedDecodeResult {
    /// Audio for the new block only (context portion discarded) [B, 1, blockFrames * 2048]
    let blockAudio: MLXArray
    /// Last `overlapSamples` from the context decode — represents the same temporal
    /// region as the previous block's held-back tail, but re-decoded with the new block's
    /// latent present. Use for true overlap-add crossfading: blending two versions of the
    /// same temporal content eliminates ghosting artifacts from adjacent-signal crossfades.
    /// Shape: [B, 1, overlapSamples], or nil if no context or overlapSamples=0.
    let contextOverlap: MLXArray?
}

/// Decode a block's latents with previous blocks prepended as causal context.
///
/// Fish S1 DAC is fully causal (left-only padding in all convolutions, windowed causal
/// attention in FishWindowLimitedTransformer with window=128). By prepending previous
/// block latents as context, the decoder sees the causal history needed for the new block.
///
/// The `maxContextFrames` parameter limits how many context frames are prepended.
/// The Fish S1 DAC decoder's effective receptive field is bounded by its causal
/// convolutions (kernel=7, dilations 1/3/9) and windowed attention. Using 8 frames
/// of context (~16,384 samples ≈ 0.37s) is sufficient for the decoder to produce
/// audio identical to full-context decode, while dramatically reducing decode time
/// (e.g., Block 1 decode drops from 3.9s to 2.6s with 8-frame context).
///
/// When `overlapSamples > 0`, also returns the last N samples of the context decode.
/// These samples represent the same temporal region as the previous block's held-back
/// tail, enabling true overlap-add crossfading (blend two decodings of the same content
/// rather than adjacent content).
///
/// - Parameters:
///   - fishAE: Fish S1 DAC codec
///   - pcaState: PCA state for inverse PCA transform
///   - contextLatent: Concatenated latents from all previous blocks [B, contextFrames, latentSize], or nil for first block
///   - blockLatent: This block's latents [B, blockFrames, latentSize]
///   - maxContextFrames: Maximum number of context frames to prepend (nil = unlimited)
///   - overlapSamples: Number of samples from the end of the context decode to return for crossfading (0 = none)
/// - Returns: `EchoOverlappedDecodeResult` with block audio and optional context overlap
func echoAEDecodeOverlapped(
    fishAE: FishS1DAC, pcaState: EchoPCAState,
    contextLatent: MLXArray?, blockLatent: MLXArray,
    maxContextFrames: Int? = nil,
    overlapSamples: Int = 0
) -> EchoOverlappedDecodeResult {
    let fullLatent: MLXArray
    let contextFrames: Int

    if let ctx = contextLatent, ctx.dim(1) > 0 {
        // Limit context to the last maxContextFrames frames (keep the most recent)
        var trimmedCtx = ctx
        if let maxFrames = maxContextFrames, ctx.dim(1) > maxFrames {
            trimmedCtx = ctx[0..., (ctx.dim(1) - maxFrames)..., 0...]
        }
        fullLatent = MLX.concatenated([trimmedCtx, blockLatent], axis: 1)
        contextFrames = trimmedCtx.dim(1)
    } else {
        fullLatent = blockLatent
        contextFrames = 0
    }

    // Decode full sequence through codec (inverse PCA + Fish S1 DAC)
    let fullAudio = echoAEDecode(fishAE: fishAE, pcaState: pcaState, zQ: fullLatent)

    if contextFrames > 0 {
        let contextSamples = contextFrames * 2048
        let blockAudio = fullAudio[0..., 0..., contextSamples...]

        // Extract overlap from the end of the context decode for crossfading.
        // This represents the same temporal region as the previous block's tail,
        // re-decoded with the new block's latent as "future" context.
        var overlap: MLXArray? = nil
        if overlapSamples > 0 && contextSamples >= overlapSamples {
            overlap = fullAudio[0..., 0..., (contextSamples - overlapSamples)..<contextSamples]
        }

        return EchoOverlappedDecodeResult(blockAudio: blockAudio, contextOverlap: overlap)
    }

    return EchoOverlappedDecodeResult(blockAudio: fullAudio, contextOverlap: nil)
}

// MARK: - Silence Detection

/// Find the point where generated latents flatten to zero (silence).
/// Input: latent [T, latentSize] (single sequence, not batched - matching Python).
/// Returns the frame index where silence starts.
func echoFindFlatteningPoint(_ latent: MLXArray, windowSize: Int = 20, stdThreshold: Float = 0.05) -> Int {
    let seqLen = latent.dim(0)
    // Pad with zeros at end (Python does this)
    let padded = MLX.concatenated([
        latent,
        MLXArray.zeros([windowSize, latent.dim(-1)])
    ], axis: 0)

    let paddedLen = padded.dim(0)
    for i in 0..<(paddedLen - windowSize) {
        let window = padded[i..<(i + windowSize)]
        let stdVal = Float(MLX.std(window).item(Float.self))
        let meanVal = abs(Float(MLX.mean(window).item(Float.self)))
        if stdVal < stdThreshold && meanVal < 0.1 {
            return i
        }
    }
    return seqLen
}

/// Crop audio at the flattening point.
/// Python: audio[..., :flattening_point * 2048]
func echoCropAudioToFlatteningPoint(audio: MLXArray, latent: MLXArray) -> MLXArray {
    let flatteningPoint = echoFindFlatteningPoint(latent)
    let cropSamples = flatteningPoint * 2048
    if cropSamples > 0 && cropSamples < audio.dim(-1) {
        return audio[0..., 0..., ..<cropSamples]
    }
    return audio
}

// MARK: - Speaker Latent Extraction

/// Extract speaker latent from reference audio.
/// Input: reference audio [1, samples] (2D, matching Python).
/// Returns (speakerLatent [1, T_latent, latentSize], mask [1, T_latent]).
func echoGetSpeakerLatentAndMask(
    fishAE: FishS1DAC, pcaState: EchoPCAState, audio: MLXArray, config: EchoTTSConfig
) -> (MLXArray, MLXArray) {
    let aeDownsampleFactor = 2048
    let maxAudioLen = config.maxSpeakerLatentLength * aeDownsampleFactor
    let audioChunkSize = config.sampler.sequenceLength * aeDownsampleFactor

    // audio: [1, samples] - limit length
    var refAudio = audio
    if refAudio.dim(1) > maxAudioLen {
        refAudio = refAudio[0..., ..<maxAudioLen]
    }

    let totalSamples = refAudio.dim(1)
    var latentArr: [MLXArray] = []

    // Process in chunks (Python: for i in range(0, audio.shape[1], audio_chunk_size))
    var i = 0
    while i < totalSamples {
        let end = min(i + audioChunkSize, totalSamples)
        var audioChunk = refAudio[0..., i..<end]

        // Pad chunk if needed
        if audioChunk.dim(1) < audioChunkSize {
            let pad = audioChunkSize - audioChunk.dim(1)
            audioChunk = MLX.padded(audioChunk, widths: [.init(0), .init((0, pad))])
        }

        // Python: ae_encode expects [B, 1, samples], so audio_chunk[:, None, :]
        let audioChunk3D = audioChunk.expandedDimensions(axis: 1)  // [1, 1, samples]
        let latentChunk = echoAEEncode(fishAE: fishAE, pcaState: pcaState, audio: audioChunk3D)
        latentArr.append(latentChunk)
        i = end
    }

    // Concatenate all latent chunks
    var speakerLatent = latentArr.isEmpty
        ? MLXArray.zeros([1, 0, 80])
        : MLX.concatenated(latentArr, axis: 1)

    // Compute actual latent length
    let actualLatentLength = totalSamples / aeDownsampleFactor

    // Create mask using arange (Python: mx.arange(speaker_latent.shape[1]) < actual_latent_length)
    let latentLen = speakerLatent.dim(1)
    let arangeArr = MLXArray(Array(0..<Int32(latentLen))).reshaped([1, latentLen])
    let speakerMask = arangeArr .< MLXArray(Int32(actualLatentLength))

    // Trim to actual length (Python: speaker_latent = speaker_latent[:, :actual_latent_length])
    speakerLatent = speakerLatent[0..., ..<actualLatentLength, 0...]
    let trimmedMask = speakerMask[0..., ..<actualLatentLength]

    // Truncate to multiple of patch_size (Python: limit = (len // patch_size) * patch_size)
    let patchSize = config.dit.speakerPatchSize
    if speakerLatent.dim(1) > 0 {
        let limit = (speakerLatent.dim(1) / patchSize) * patchSize
        if limit > 0 {
            return (speakerLatent[0..., ..<limit, 0...], trimmedMask[0..., ..<limit])
        }
    }

    return (speakerLatent, trimmedMask)
}
