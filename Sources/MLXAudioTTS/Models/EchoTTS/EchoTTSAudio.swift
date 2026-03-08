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
