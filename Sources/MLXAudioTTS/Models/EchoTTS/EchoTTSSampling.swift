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
///
/// Each block is a separate diffusion run over `numSteps`. Previous blocks' latents
/// are encoded as a latent KV cache for temporal coherence. The `onBlockComplete`
/// callback fires after each block's diffusion loop with:
///   - `blockLatent`: this block's completed latent [B, blockSize, latentSize]
///   - `contextLatent`: concatenated previous blocks [B, prevFrames, latentSize] (nil for first block)
///
/// The callback returns `true` to continue generating the next block, or `false`
/// to stop early (e.g. when silence is detected and remaining blocks would be empty).
///
/// Speaker KV scaling is applied fresh at the start of each block and un-done
/// at the crossover timestep within each block's inner loop.
///
/// This overload accepts pre-computed text/speaker KV caches to remove them from
/// the TTFA critical path (compute them before calling this function).
func echoSampleBlockwiseEulerCFG(
    model: EchoDiT,
    textKVCond: EchoKVCache,
    speakerKVCondBase: EchoKVCache,
    textMask: MLXArray,
    speakerMask: MLXArray,
    blockSizes: [Int],
    config: EchoTTSConfig,
    rngSeed: UInt64? = nil,
    onBlockComplete: ((_ blockLatent: MLXArray, _ contextLatent: MLXArray?) -> Bool)? = nil
) -> MLXArray {
    let samplerConfig = config.sampler
    let ditConfig = config.dit
    let batchSize = textMask.dim(0)
    let latentSize = ditConfig.latentSize
    let numSteps = samplerConfig.numSteps

    if let seed = rngSeed {
        MLXRandom.seed(seed)
    }

    // Time schedule (same for every block): linspace(0.999, 0, numSteps+1)
    var tSchedule = [Float](repeating: 0, count: numSteps + 1)
    for i in 0...numSteps {
        tSchedule[i] = 0.999 * (1.0 - Float(i) / Float(numSteps))
    }

    // Unconditional masks (zeros = don't attend)
    let textMaskUncond = MLXArray.zeros(like: textMask)
    let speakerMaskUncond = MLXArray.zeros(like: speakerMask)

    // Triple text KV cache (constant across blocks, already pre-computed)
    let textKVFull = echoConcatKVCaches([textKVCond, textKVCond, textKVCond])

    // Triple masks: [cond_text, uncond_text, cond_text] and [cond_spk, cond_spk, uncond_spk]
    let fullTextMask = MLX.concatenated([textMask, textMaskUncond, textMask], axis: 0)
    let fullSpeakerMask = MLX.concatenated([speakerMask, speakerMask, speakerMaskUncond], axis: 0)

    // Speaker KV starts from the base (pre-computed, before any scaling)
    var speakerKVCond = speakerKVCondBase

    var generatedChunks: [MLXArray] = []
    var startPos = 0

    for (blockIdx, blockSize) in blockSizes.enumerated() {
        let blockStart = Date()

        // Apply speaker KV scaling at the start of each block (Python: per-block)
        // Each block's crossover will un-do this, so next block re-applies.
        if let kvScale = samplerConfig.speakerKvScale {
            speakerKVCond = echoMultiplyKVCache(
                speakerKVCond, scale: kvScale,
                maxLayers: samplerConfig.speakerKvMaxLayers
            )
        }
        var speakerKVFull = echoConcatKVCaches([speakerKVCond, speakerKVCond, speakerKVCond])

        // Compute latent KV cache from previously generated blocks
        let prefixLatent: MLXArray
        if generatedChunks.isEmpty {
            prefixLatent = MLXArray.zeros([batchSize, 0, latentSize])
        } else {
            prefixLatent = MLX.concatenated(generatedChunks, axis: 1)
        }

        // Triple prefix for CFG (all 3 branches see same latent prefix)
        let fullPrefixLatent = MLX.concatenated([prefixLatent, prefixLatent, prefixLatent], axis: 0)
        let latentKVFull = model.getKVCacheLatent(fullPrefixLatent)
        let latentKVCond: EchoKVCache = latentKVFull.map { (k, v) in
            (k[..<batchSize], v[..<batchSize])
        }

        // Initialize noise for this block
        var xT = MLXRandom.normal([batchSize, blockSize, latentSize]) * samplerConfig.truncationFactor

        // Euler integration for this block
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
                    latentKVCache: latentKVFull,
                    textMask: fullTextMask,
                    speakerMask: fullSpeakerMask,
                    startPos: startPos
                )

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
                    latentKVCache: latentKVCond,
                    textMask: textMask,
                    speakerMask: speakerMask,
                    startPos: startPos
                )
            }

            // Temporal rescaling
            if let rescaleK = samplerConfig.rescaleK,
               let rescaleSigma = samplerConfig.rescaleSigma {
                vPred = echoTemporalScoreRescale(vPred, t: t, rescaleK: rescaleK, rescaleSigma: rescaleSigma)
            }

            // Speaker KV scale crossover (un-do scaling at boundary)
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

        let blockTime = Date().timeIntervalSince(blockStart)
        print("[EchoTTS] Block \(blockIdx) (\(blockSize) frames): diffusion completed in \(String(format: "%.2f", blockTime))s")

        // Build context from previous blocks (nil for first block)
        let contextLatent: MLXArray? = generatedChunks.isEmpty
            ? nil
            : MLX.concatenated(generatedChunks, axis: 1)

        generatedChunks.append(xT)
        startPos += blockSize

        // Notify callback with this block's latent and previous context
        // Returns false to stop early (e.g. remaining blocks are silence)
        let shouldContinue = onBlockComplete?(xT, contextLatent) ?? true
        if !shouldContinue {
            print("[EchoTTS] Early termination after block \(blockIdx) — remaining blocks skipped")
            break
        }
    }

    return MLX.concatenated(generatedChunks, axis: 1)
}
