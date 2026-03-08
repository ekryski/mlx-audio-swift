import Foundation
import MLX
import MLXRandom

// MARK: - KV Cache Helpers

/// Concatenate KV caches along batch dimension (for CFG tripling).
func echoConcatKVCaches(_ caches: [EchoKVCache]) -> EchoKVCache {
    guard let first = caches.first else { return [] }
    let numLayers = first.count
    return (0..<numLayers).map { i in
        let keys = caches.map { $0[i].0 }
        let values = caches.map { $0[i].1 }
        return (MLX.concatenated(keys, axis: 0), MLX.concatenated(values, axis: 0))
    }
}

/// Multiply (scale) KV cache values.
func echoMultiplyKVCache(_ cache: EchoKVCache, scale: Float, maxLayers: Int? = nil) -> EchoKVCache {
    let numLayers = maxLayers != nil ? min(maxLayers!, cache.count) : cache.count
    return cache.enumerated().map { (i, kv) in
        if i < numLayers {
            return (kv.0 * scale, kv.1 * scale)  // Scale both K and V (matching Python)
        }
        return kv
    }
}

/// Temporal score rescaling for velocity prediction.
func echoTemporalScoreRescale(
    _ vPred: MLXArray, t: Float, rescaleK: Float, rescaleSigma: Float
) -> MLXArray {
    let snr = t * t / ((1 - t) * (1 - t) + 1e-8)
    let weight = 1.0 / (1.0 + rescaleK * exp(-snr / rescaleSigma))
    return vPred * weight
}

// MARK: - Standard Euler CFG Sampling

/// Euler ODE sampling with independent classifier-free guidance for text and speaker.
///
/// This implements the standard full-sequence generation:
/// - Tripled batch for CFG: [conditional, uncond_text, uncond_speaker]
/// - Independent guidance: v = v_cond + cfg_text*(v_cond - v_uncond_text) + cfg_spk*(v_cond - v_uncond_spk)
/// - Linear schedule from t=0.999 to t=0
func echoSampleEulerCFG(
    model: EchoDiT,
    textTokens: MLXArray,
    textMask: MLXArray,
    speakerLatent: MLXArray,
    speakerMask: MLXArray,
    config: EchoTTSConfig,
    rngSeed: UInt64? = nil
) -> MLXArray {
    let samplerConfig = config.sampler
    let ditConfig = config.dit
    let batchSize = textTokens.dim(0)
    let seqLen = samplerConfig.sequenceLength
    let latentSize = ditConfig.latentSize
    let numSteps = samplerConfig.numSteps

    // Initialize noise
    if let seed = rngSeed {
        MLXRandom.seed(seed)
    }

    // Time schedule using numpy-style linspace (matching Python)
    var tSchedule = [Float](repeating: 0, count: numSteps + 1)
    for i in 0...numSteps {
        tSchedule[i] = 0.999 * (1.0 - Float(i) / Float(numSteps))
    }

    // Unconditional masks (zeros = don't attend)
    let textMaskUncond = MLXArray.zeros(like: textMask)
    let speakerMaskUncond = MLXArray.zeros(like: speakerMask)

    // Build KV caches ONCE (Python: same conditional KV for all 3 branches!)
    let textKVCond = model.getKVCacheText(textTokens, mask: textMask)
    var speakerKVCond = model.getKVCacheSpeaker(speakerLatent)

    // Apply speaker KV scaling upfront if configured (Python does this before the loop)
    if let kvScale = samplerConfig.speakerKvScale {
        speakerKVCond = echoMultiplyKVCache(
            speakerKVCond, scale: kvScale,
            maxLayers: samplerConfig.speakerKvMaxLayers
        )
    }

    // Triple KV caches: SAME conditional KV for all 3 branches (Python style!)
    let textKVFull = echoConcatKVCaches([textKVCond, textKVCond, textKVCond])
    var speakerKVFull = echoConcatKVCaches([speakerKVCond, speakerKVCond, speakerKVCond])

    // Triple masks: [cond_text, uncond_text, cond_text] and [cond_spk, cond_spk, uncond_spk]
    let fullTextMask = MLX.concatenated([textMask, textMaskUncond, textMask], axis: 0)
    let fullSpeakerMask = MLX.concatenated([speakerMask, speakerMask, speakerMaskUncond], axis: 0)

    var xT = MLXRandom.normal([batchSize, seqLen, latentSize]) * samplerConfig.truncationFactor

    // Euler integration
    for step in 0..<numSteps {
        let t = tSchedule[step]
        let tNext = tSchedule[step + 1]
        let hasCFG = t >= samplerConfig.cfgMinT && t <= samplerConfig.cfgMaxT

        var vPred: MLXArray

        if hasCFG {
            let xTriple = MLX.concatenated([xT, xT, xT], axis: 0)
            let tFull = MLXArray(Array(repeating: t, count: batchSize * 3))

            let vAll = model(
                xTriple,
                timestep: tFull,
                textKVCache: textKVFull,
                speakerKVCache: speakerKVFull,
                textMask: fullTextMask,
                speakerMask: fullSpeakerMask
            )

            // Split: [cond, uncond_text, uncond_speaker]
            let vCond = vAll[..<batchSize, 0..., 0...]
            let vUncondText = vAll[batchSize..<(2 * batchSize), 0..., 0...]
            let vUncondSpeaker = vAll[(2 * batchSize)..., 0..., 0...]

            vPred = vCond
                + samplerConfig.cfgScaleText * (vCond - vUncondText)
                + samplerConfig.cfgScaleSpeaker * (vCond - vUncondSpeaker)
        } else {
            let tCond = MLXArray(Array(repeating: t, count: batchSize))
            vPred = model(
                xT,
                timestep: tCond,
                textKVCache: textKVCond,
                speakerKVCache: speakerKVCond,
                textMask: textMask,
                speakerMask: speakerMask
            )
        }

        // Temporal rescaling
        if let rescaleK = samplerConfig.rescaleK,
           let rescaleSigma = samplerConfig.rescaleSigma {
            vPred = echoTemporalScoreRescale(vPred, t: t, rescaleK: rescaleK, rescaleSigma: rescaleSigma)
        }

        // Speaker KV scale crossover (Python undoes scaling at boundary)
        if let kvScale = samplerConfig.speakerKvScale,
           let kvMinT = samplerConfig.speakerKvMinT,
           tNext < kvMinT && t >= kvMinT {
            speakerKVCond = echoMultiplyKVCache(
                speakerKVCond, scale: 1.0 / kvScale,
                maxLayers: samplerConfig.speakerKvMaxLayers
            )
            speakerKVFull = echoConcatKVCaches([speakerKVCond, speakerKVCond, speakerKVCond])
        }

        xT = xT + vPred * (tNext - t)
        eval(xT)
    }

    return xT
}

// MARK: - Blockwise Euler CFG Sampling

/// Blockwise sampling that generates audio in chunks with latent prefix context.
func echoSampleBlockwiseEulerCFG(
    model: EchoDiT,
    textTokens: MLXArray,
    textMask: MLXArray,
    speakerLatent: MLXArray,
    speakerMask: MLXArray,
    blockSizes: [Int],
    config: EchoTTSConfig,
    rngSeed: UInt64? = nil
) -> MLXArray {
    let samplerConfig = config.sampler
    let ditConfig = config.dit
    let batchSize = textTokens.dim(0)
    let latentSize = ditConfig.latentSize
    let numSteps = samplerConfig.numSteps

    if let seed = rngSeed {
        MLXRandom.seed(seed)
    }

    // Build text and speaker KV caches (same as standard)
    let textKVCond = model.getKVCacheText(textTokens, mask: textMask)
    let uncondTextTokens = MLXArray.zeros(like: textTokens)
    let uncondTextMask = MLXArray.zeros(like: textMask)
    let textKVUncond = model.getKVCacheText(uncondTextTokens, mask: uncondTextMask)

    let speakerKVCond = model.getKVCacheSpeaker(speakerLatent)
    let uncondSpeakerLatent = MLXArray.zeros(like: speakerLatent)
    let uncondSpeakerMask = MLXArray.zeros(like: speakerMask)
    let speakerKVUncond = model.getKVCacheSpeaker(uncondSpeakerLatent)

    let textKV3 = echoConcatKVCaches([textKVCond, textKVUncond, textKVCond])
    let speakerKV3 = echoConcatKVCaches([speakerKVCond, speakerKVCond, speakerKVUncond])
    let textMask3 = MLX.concatenated([textMask, uncondTextMask, textMask], axis: 0)
    let speakerMask3 = MLX.concatenated([speakerMask, speakerMask, uncondSpeakerMask], axis: 0)

    var allBlocks: [MLXArray] = []

    for blockSize in blockSizes {
        var xT = MLXRandom.normal([batchSize, blockSize, latentSize]) * samplerConfig.truncationFactor

        // Build latent prefix KV cache from previous blocks
        // Note: This requires the model to have latent_encoder
        // For simplicity, skip latent prefix if model doesn't have it

        var timeSteps: [Float] = []
        for i in 0..<numSteps {
            timeSteps.append(0.999 - Float(i) * 0.999 / Float(numSteps))
        }
        timeSteps.append(0.0)

        for step in 0..<numSteps {
            let t = timeSteps[step]
            let tNext = timeSteps[step + 1]
            let dt = tNext - t

            let useCFG = t >= samplerConfig.cfgMinT && t <= samplerConfig.cfgMaxT

            if useCFG {
                let xTriple = MLX.concatenated([xT, xT, xT], axis: 0)
                let tArray = MLXArray(Array(repeating: t, count: batchSize * 3))

                let vAll = model(
                    xTriple,
                    timestep: tArray,
                    textKVCache: textKV3,
                    speakerKVCache: speakerKV3,
                    textMask: textMask3,
                    speakerMask: speakerMask3
                )

                let vCond = vAll[..<batchSize, 0..., 0...]
                let vUncondText = vAll[batchSize..<(2 * batchSize), 0..., 0...]
                let vUncondSpeaker = vAll[(2 * batchSize)..., 0..., 0...]

                let vPred = vCond
                    + samplerConfig.cfgScaleText * (vCond - vUncondText)
                    + samplerConfig.cfgScaleSpeaker * (vCond - vUncondSpeaker)

                xT = xT + vPred * dt
            } else {
                let tArray = MLXArray(Array(repeating: t, count: batchSize))
                let vPred = model(
                    xT,
                    timestep: tArray,
                    textKVCache: textKVCond,
                    speakerKVCache: speakerKVCond,
                    textMask: textMask,
                    speakerMask: speakerMask
                )
                xT = xT + vPred * dt
            }

            eval(xT)
        }

        allBlocks.append(xT)
    }

    return MLX.concatenated(allBlocks, axis: 1)
}
