import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - RoPE Helpers

func echoPrecomputeFreqsCis(dim: Int, end: Int, theta: Float = 10000.0) -> (MLXArray, MLXArray) {
    let halfDim = dim / 2
    var freqs = [Float](repeating: 0, count: halfDim)
    for i in 0..<halfDim {
        freqs[i] = 1.0 / pow(theta, Float(2 * i) / Float(dim))
    }
    let freqsArray = MLXArray(freqs)

    var positions = [Float](repeating: 0, count: end)
    for i in 0..<end {
        positions[i] = Float(i)
    }
    let posArray = MLXArray(positions)

    // [end, halfDim]
    let angles = posArray.reshaped([end, 1]) * freqsArray.reshaped([1, halfDim])
    return (MLX.cos(angles), MLX.sin(angles))
}

func echoApplyRotaryEmb(_ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray) -> MLXArray {
    // x: [B, seqLen, nHeads, headDim] (Python layout, before SDPA transpose)
    // cosFreqs/sinFreqs: [seqLen, halfDim]
    // Uses interleaved even/odd indices (matching Python implementation)
    let xEven = gatherAlternate(x, even: true)   // x[..., 0::2]
    let xOdd = gatherAlternate(x, even: false)    // x[..., 1::2]

    // Reshape cos/sin for broadcasting: [1, seqLen, 1, halfDim]
    let cos = cosFreqs.reshaped([1, cosFreqs.dim(0), 1, cosFreqs.dim(1)])
    let sin = sinFreqs.reshaped([1, sinFreqs.dim(0), 1, sinFreqs.dim(1)])

    let rotEven = xEven * cos - xOdd * sin
    let rotOdd = xOdd * cos + xEven * sin

    // Interleave back: stack along last axis then reshape
    let stacked = MLX.stacked([rotEven, rotOdd], axis: -1)  // [..., halfDim, 2]
    return stacked.reshaped(x.shape)
}

/// Gather even or odd indices along the last axis: x[..., 0::2] or x[..., 1::2]
func gatherAlternate(_ x: MLXArray, even: Bool) -> MLXArray {
    let lastDim = x.dim(-1)
    let halfDim = lastDim / 2
    let start = even ? 0 : 1
    // Use stride-based indexing
    var indices = [Int32](repeating: 0, count: halfDim)
    for i in 0..<halfDim {
        indices[i] = Int32(start + i * 2)
    }
    let idxArray = MLXArray(indices)
    return x.take(idxArray, axis: x.ndim - 1)
}

// MARK: - Timestep Embedding

func echoGetTimestepEmbedding(_ timestep: MLXArray, embedSize: Int) -> MLXArray {
    let halfDim = embedSize / 2
    let base = log(Float(10000.0))
    var freqs = [Float](repeating: 0, count: halfDim)
    for i in 0..<halfDim {
        freqs[i] = 1000.0 * exp(-base * Float(i) / Float(halfDim))
    }
    let freqsArray = MLXArray(freqs)

    // timestep: [B] or [B, 1]
    let t = timestep.reshaped([-1, 1])  // [B, 1]
    let args = t * freqsArray.reshaped([1, halfDim])  // [B, halfDim]

    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)  // [B, embedSize]
}

// MARK: - Mask Helpers

func echoBoolToAdditiveMask(_ mask: MLXArray) -> MLXArray {
    // Convert boolean mask (true=attend) to additive mask (0=attend, -1e9=ignore)
    // Python: mx.where(mask, zero, neg_inf)[:, None, :, :]
    let zero = MLXArray.zeros(mask.shape)
    let negInf = MLXArray.full(mask.shape, values: MLXArray(-1e9))
    let result = MLX.where(mask, zero, negInf)
    // Add head dimension: [B, seqLen, kvLen] -> [B, 1, seqLen, kvLen]
    if result.ndim == 3 {
        return result.expandedDimensions(axis: 1)
    }
    return result
}

func echoMakeCausalMask(seqLen: Int) -> MLXArray {
    var data = [Float](repeating: -Float.infinity, count: seqLen * seqLen)
    for i in 0..<seqLen {
        for j in 0...i {
            data[i * seqLen + j] = 0.0
        }
    }
    return MLXArray(data, [seqLen, seqLen])
}

/// Boolean causal mask: row >= col
func echoMakeCausalMaskBool(seqLen: Int) -> MLXArray {
    let row = MLXArray(Array(0..<Int32(seqLen))).reshaped([seqLen, 1])
    let col = MLXArray(Array(0..<Int32(seqLen))).reshaped([1, seqLen])
    return row .>= col  // [seqLen, seqLen]
}

// MARK: - RMS Normalization

/// RMS normalization supporting both scalar and per-head dimensions.
class EchoRMSNorm: Module {
    let eps: Float
    let perHead: Bool

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.perHead = false
        self._weight.wrappedValue = MLXArray.ones([dim])
    }

    init(numHeads: Int, headDim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.perHead = true
        self._weight.wrappedValue = MLXArray.ones([numHeads, headDim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Python: cast to float32, use rsqrt, cast back
        let xDtype = x.dtype
        let xFloat = x.asType(.float32)
        let rms = MLX.sqrt(MLX.mean(xFloat * xFloat, axis: -1, keepDims: true) + eps)
        let result = (xFloat / rms) * weight
        return result.asType(xDtype)
    }
}

// MARK: - Low-Rank Adaptive Layer Normalization

/// Adaptive layer normalization with low-rank shift/scale/gate decomposition.
/// Python: LowRankAdaLN - applies silu before down projection, residual is simply + input.
class EchoLowRankAdaLN: Module {
    let modelSize: Int
    let rank: Int
    let eps: Float

    @ModuleInfo(key: "shift_down") var shiftDown: Linear
    @ModuleInfo(key: "shift_up") var shiftUp: Linear
    @ModuleInfo(key: "scale_down") var scaleDown: Linear
    @ModuleInfo(key: "scale_up") var scaleUp: Linear
    @ModuleInfo(key: "gate_down") var gateDown: Linear
    @ModuleInfo(key: "gate_up") var gateUp: Linear

    init(modelSize: Int, rank: Int, eps: Float = 1e-5) {
        self.modelSize = modelSize
        self.rank = rank
        self.eps = eps

        // Python: nn.Linear(model_size, rank, bias=False) for down, nn.Linear(rank, model_size, bias=True) for up
        self._shiftDown.wrappedValue = Linear(modelSize, rank, bias: false)
        self._shiftUp.wrappedValue = Linear(rank, modelSize)  // bias=true (default)
        self._scaleDown.wrappedValue = Linear(modelSize, rank, bias: false)
        self._scaleUp.wrappedValue = Linear(rank, modelSize)  // bias=true
        self._gateDown.wrappedValue = Linear(modelSize, rank, bias: false)
        self._gateUp.wrappedValue = Linear(rank, modelSize)  // bias=true
    }

    /// Returns (normalized_x, gate).
    func callAsFunction(_ x: MLXArray, condEmbed: MLXArray) -> (MLXArray, MLXArray) {
        // condEmbed: [B, 1, condSize] split into 3 parts
        let condSize = condEmbed.dim(-1) / 3
        var shiftVal = condEmbed[0..., 0..., ..<condSize]
        var scaleVal = condEmbed[0..., 0..., condSize..<(2 * condSize)]
        var gateVal = condEmbed[0..., 0..., (2 * condSize)...]

        // Python: shift = self.shift_up(self.shift_down(nn.silu(shift))) + shift
        shiftVal = shiftUp(shiftDown(silu(shiftVal))) + shiftVal
        scaleVal = scaleUp(scaleDown(silu(scaleVal))) + scaleVal
        gateVal = gateUp(gateDown(silu(gateVal))) + gateVal

        // RMS norm inline (Python: keeps in float32 through scale/shift, casts back at the end)
        let xDtype = x.dtype
        let xFloat = x.asType(.float32)
        let xNorm = xFloat * MLX.rsqrt(MLX.mean(xFloat * xFloat, axis: -1, keepDims: true) + eps)

        // Apply scale/shift in float32 space, then cast back (matching Python)
        let normalized = (xNorm * (scaleVal + 1.0) + shiftVal).asType(xDtype)
        let gate = MLX.tanh(gateVal)
        return (normalized, gate)
    }
}

// MARK: - Self Attention (for Text/Speaker Encoders)

class EchoSelfAttention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "q_norm") var qNorm: EchoRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EchoRMSNorm

    init(modelSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = modelSize / numHeads

        self._wq.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._wk.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._wv.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._wo.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._gate.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._qNorm.wrappedValue = EchoRMSNorm(numHeads: numHeads, headDim: headDim)
        self._kNorm.wrappedValue = EchoRMSNorm(numHeads: numHeads, headDim: headDim)
    }

    func callAsFunction(
        _ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Keep in [B, seqLen, numHeads, headDim] for QK norm and RoPE (matching Python)
        var q = wq(x).reshaped([bsz, seqLen, numHeads, headDim])
        var k = wk(x).reshaped([bsz, seqLen, numHeads, headDim])
        let v = wv(x).reshaped([bsz, seqLen, numHeads, headDim])
        let gateVal = gate(x)

        // QK normalization (works on [B, seqLen, numHeads, headDim])
        q = qNorm(q)
        k = kNorm(k)

        // RoPE applied in [B, seqLen, numHeads, headDim] layout
        let cosSliced = cosFreqs[..<seqLen]
        let sinSliced = sinFreqs[..<seqLen]
        q = echoApplyRotaryEmb(q, cosFreqs: cosSliced, sinFreqs: sinSliced)
        k = echoApplyRotaryEmb(k, cosFreqs: cosSliced, sinFreqs: sinSliced)

        // Now transpose for SDPA: [B, numHeads, seqLen, headDim]
        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        let scale = sqrt(Float(headDim))
        let output = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kT, values: vT,
            scale: 1.0 / scale, mask: mask
        )

        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([bsz, seqLen, numHeads * headDim])

        // Gated output
        return wo(outputReshaped * sigmoid(gateVal))
    }
}

// MARK: - Joint Attention (for Main DiT Blocks)

/// Typealias for KV cache: list of (key, value) pairs per layer.
public typealias EchoKVCache = [(MLXArray, MLXArray)]

class EchoJointAttention: Module {
    let numHeads: Int
    let headDim: Int
    let modelSize: Int

    // Self-attention projections
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "gate") var gate: Linear

    // Cross-attention to text
    @ModuleInfo(key: "wk_text") var wkText: Linear
    @ModuleInfo(key: "wv_text") var wvText: Linear

    // Cross-attention to speaker
    @ModuleInfo(key: "wk_speaker") var wkSpeaker: Linear
    @ModuleInfo(key: "wv_speaker") var wvSpeaker: Linear

    // Optional cross-attention to latent prefix (blockwise)
    @ModuleInfo(key: "wk_latent") var wkLatent: Linear?
    @ModuleInfo(key: "wv_latent") var wvLatent: Linear?

    // QK normalization
    @ModuleInfo(key: "q_norm") var qNorm: EchoRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EchoRMSNorm

    init(config: EchoDiTConfig, hasLatentAttention: Bool = false) {
        self.numHeads = config.numHeads
        self.headDim = config.modelSize / config.numHeads
        self.modelSize = config.modelSize

        self._wq.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._wk.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._wv.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._wo.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._gate.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)

        self._wkText.wrappedValue = Linear(config.textModelSize, config.modelSize, bias: false)
        self._wvText.wrappedValue = Linear(config.textModelSize, config.modelSize, bias: false)

        self._wkSpeaker.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)
        self._wvSpeaker.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)

        if hasLatentAttention {
            self._wkLatent.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)
            self._wvLatent.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)
        } else {
            self._wkLatent.wrappedValue = nil
            self._wvLatent.wrappedValue = nil
        }

        self._qNorm.wrappedValue = EchoRMSNorm(numHeads: config.numHeads, headDim: headDim)
        self._kNorm.wrappedValue = EchoRMSNorm(numHeads: config.numHeads, headDim: headDim)
    }

    /// Build text KV cache for a layer. Returns [B, seqLen, numHeads, headDim].
    func getKVCacheText(_ textState: MLXArray) -> (MLXArray, MLXArray) {
        let bsz = textState.dim(0)
        let seqLen = textState.dim(1)
        let k = kNorm(wkText(textState).reshaped([bsz, seqLen, numHeads, headDim]))
        let v = wvText(textState).reshaped([bsz, seqLen, numHeads, headDim])
        return (k, v)
    }

    /// Build speaker KV cache for a layer. Returns [B, seqLen, numHeads, headDim].
    func getKVCacheSpeaker(_ speakerState: MLXArray) -> (MLXArray, MLXArray) {
        let bsz = speakerState.dim(0)
        let seqLen = speakerState.dim(1)
        let k = kNorm(wkSpeaker(speakerState).reshaped([bsz, seqLen, numHeads, headDim]))
        let v = wvSpeaker(speakerState).reshaped([bsz, seqLen, numHeads, headDim])
        return (k, v)
    }

    /// Build latent prefix KV cache for a layer (blockwise generation).
    /// Unlike text/speaker KV, latent keys also get half-head RoPE applied.
    /// Returns [B, seqLen, numHeads, headDim].
    func getKVCacheLatent(
        _ latentState: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray
    ) -> (MLXArray, MLXArray) {
        guard let wkLatent, let wvLatent else {
            fatalError("Latent KV modules not loaded. Use delete_blockwise_modules=false.")
        }
        let bsz = latentState.dim(0)
        let seqLen = latentState.dim(1)
        var k = wkLatent(latentState).reshaped([bsz, seqLen, numHeads, headDim])
        let v = wvLatent(latentState).reshaped([bsz, seqLen, numHeads, headDim])
        k = kNorm(k)
        k = applyRotaryHalf(k, cosFreqs: cosFreqs, sinFreqs: sinFreqs)
        return (k, v)
    }

    /// Apply rotary embeddings to only the first half of heads (Python: _apply_rotary_half).
    /// Input shape: [B, seqLen, numHeads, headDim]
    func applyRotaryHalf(_ y: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray) -> MLXArray {
        let halfHeads = y.dim(-2) / 2  // Split on HEADS dimension, not headDim!
        let y1 = y[0..., 0..., ..<halfHeads, 0...]  // First half of heads
        let y2 = y[0..., 0..., halfHeads..., 0...]   // Second half of heads
        let y1Rotated = echoApplyRotaryEmb(y1, cosFreqs: cosFreqs, sinFreqs: sinFreqs)
        return MLX.concatenated([y1Rotated, y2], axis: -2)  // Concat along heads dim
    }

    func callAsFunction(
        _ x: MLXArray,
        cosFreqs: MLXArray, sinFreqs: MLXArray,
        startPos: Int,
        textKV: (MLXArray, MLXArray),
        speakerKV: (MLXArray, MLXArray),
        latentKV: (MLXArray, MLXArray)? = nil,
        textMask: MLXArray? = nil,
        speakerMask: MLXArray? = nil
    ) -> MLXArray {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Keep in [B, seqLen, numHeads, headDim] for QK norm and RoPE
        var q = wq(x).reshaped([bsz, seqLen, numHeads, headDim])
        var selfK = wk(x).reshaped([bsz, seqLen, numHeads, headDim])
        let selfV = wv(x).reshaped([bsz, seqLen, numHeads, headDim])
        let gateVal = gate(x)

        // QK norm
        q = qNorm(q)
        selfK = kNorm(selfK)

        // Apply RoPE to first half of HEADS (not head_dim!)
        let qCos = cosFreqs[startPos..<(startPos + seqLen)]
        let qSin = sinFreqs[startPos..<(startPos + seqLen)]
        q = applyRotaryHalf(q, cosFreqs: qCos, sinFreqs: qSin)
        selfK = applyRotaryHalf(selfK, cosFreqs: qCos, sinFreqs: qSin)

        // Transpose for SDPA and KV concatenation: [B, numHeads, seqLen, headDim]
        let qT = q.transposed(0, 2, 1, 3)
        let selfKT = selfK.transposed(0, 2, 1, 3)
        let selfVT = selfV.transposed(0, 2, 1, 3)

        // KV caches are already in [B, seqLen, numHeads, headDim] from get_kv_cache_*
        // Need to transpose them to [B, numHeads, seqLen, headDim] for SDPA
        let (textK, textV) = textKV
        let (spkK, spkV) = speakerKV

        // Concatenate all KV sources along sequence dimension: [self, latent?, text, speaker]
        var allK = [selfKT]
        var allV = [selfVT]

        if let (lk, lv) = latentKV, lk.dim(1) > 0 {
            allK.append(lk.transposed(0, 2, 1, 3))
            allV.append(lv.transposed(0, 2, 1, 3))
        }

        allK.append(textK.transposed(0, 2, 1, 3))
        allV.append(textV.transposed(0, 2, 1, 3))

        allK.append(spkK.transposed(0, 2, 1, 3))
        allV.append(spkV.transposed(0, 2, 1, 3))

        let kCat = MLX.concatenated(allK, axis: 2)
        let vCat = MLX.concatenated(allV, axis: 2)

        // Build combined attention mask (Python style: boolean mask then additive)
        // Python: mask = mx.concatenate([self_mask, latent_mask, text_mask, speaker_mask], axis=1)
        // then broadcasts [B, totalKVLen] -> [B, seqLen, totalKVLen] -> _bool_to_additive_mask
        let selfMaskBool = MLXArray.ones([bsz, seqLen], type: Bool.self)
        var maskParts: [MLXArray] = [selfMaskBool]

        // Latent mask: attend to ALL latent prefix positions (Python: mx.ones)
        if let (lk, _) = latentKV, lk.dim(1) > 0 {
            let latentLen = lk.dim(1)
            let latentMask = MLXArray.ones([bsz, latentLen], type: Bool.self)
            maskParts.append(latentMask)
        }

        // Text mask
        if let tm = textMask {
            maskParts.append(tm.asType(Bool.self))
        } else {
            maskParts.append(MLXArray.ones([bsz, textK.dim(1)], type: Bool.self))
        }

        // Speaker mask
        if let sm = speakerMask {
            maskParts.append(sm.asType(Bool.self))
        } else {
            maskParts.append(MLXArray.ones([bsz, spkK.dim(1)], type: Bool.self))
        }

        let combinedBoolMask = MLX.concatenated(maskParts, axis: 1)
        // All query positions see the same key mask, so reshape to [B, 1, 1, totalKVLen]
        // for broadcasting over heads and queries in SDPA
        let floatMask = combinedBoolMask.asType(.float32)
        let additiveMask = ((1.0 - floatMask) * (-1e9)).reshaped([bsz, 1, 1, -1])

        let scale = sqrt(Float(headDim))
        let output = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kCat, values: vCat,
            scale: 1.0 / scale, mask: additiveMask
        )

        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([bsz, seqLen, numHeads * headDim])
        return wo(outputReshaped * sigmoid(gateVal))
    }
}

// MARK: - MLP (SwiGLU)

class EchoMLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(modelSize: Int, intermediateSize: Int) {
        self._w1.wrappedValue = Linear(modelSize, intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(intermediateSize, modelSize, bias: false)
        self._w3.wrappedValue = Linear(modelSize, intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Encoder Transformer Block (for Text/Speaker Encoders)

class EchoEncoderTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: EchoSelfAttention
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "attention_norm") var attentionNorm: EchoRMSNorm
    @ModuleInfo(key: "mlp_norm") var mlpNorm: EchoRMSNorm

    init(modelSize: Int, numHeads: Int, intermediateSize: Int) {
        self._attention.wrappedValue = EchoSelfAttention(modelSize: modelSize, numHeads: numHeads)
        self._mlp.wrappedValue = EchoMLP(modelSize: modelSize, intermediateSize: intermediateSize)
        self._attentionNorm.wrappedValue = EchoRMSNorm(dim: modelSize)
        self._mlpNorm.wrappedValue = EchoRMSNorm(dim: modelSize)
    }

    func callAsFunction(
        _ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var h = x + attention(attentionNorm(x), cosFreqs: cosFreqs, sinFreqs: sinFreqs, mask: mask)
        h = h + mlp(mlpNorm(h))
        return h
    }
}

// MARK: - Main DiT Transformer Block

class EchoTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: EchoJointAttention
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "attention_adaln") var attentionAdaLN: EchoLowRankAdaLN
    @ModuleInfo(key: "mlp_adaln") var mlpAdaLN: EchoLowRankAdaLN

    init(config: EchoDiTConfig, hasLatentAttention: Bool = false) {
        self._attention.wrappedValue = EchoJointAttention(config: config, hasLatentAttention: hasLatentAttention)
        self._mlp.wrappedValue = EchoMLP(modelSize: config.modelSize, intermediateSize: config.intermediateSize)
        self._attentionAdaLN.wrappedValue = EchoLowRankAdaLN(
            modelSize: config.modelSize, rank: config.adalnRank, eps: config.normEps
        )
        self._mlpAdaLN.wrappedValue = EchoLowRankAdaLN(
            modelSize: config.modelSize, rank: config.adalnRank, eps: config.normEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        condEmbed: MLXArray,
        cosFreqs: MLXArray, sinFreqs: MLXArray,
        startPos: Int,
        textKV: (MLXArray, MLXArray),
        speakerKV: (MLXArray, MLXArray),
        latentKV: (MLXArray, MLXArray)? = nil,
        textMask: MLXArray? = nil,
        speakerMask: MLXArray? = nil
    ) -> MLXArray {
        // Attention with AdaLN
        let (xNormAttn, attnGate) = attentionAdaLN(x, condEmbed: condEmbed)
        let attnOutput = attention(
            xNormAttn,
            cosFreqs: cosFreqs, sinFreqs: sinFreqs,
            startPos: startPos,
            textKV: textKV, speakerKV: speakerKV,
            latentKV: latentKV,
            textMask: textMask, speakerMask: speakerMask
        )
        var h = x + attnGate * attnOutput

        // MLP with AdaLN
        let (hNormMLP, mlpGate) = mlpAdaLN(h, condEmbed: condEmbed)
        h = h + mlpGate * mlp(hNormMLP)

        return h
    }
}

// MARK: - Text Encoder

class EchoTextEncoder: Module {
    let modelSize: Int
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "text_embedding") var embedding: Embedding
    @ModuleInfo(key: "blocks") var layers: [EchoEncoderTransformerBlock]

    init(config: EchoDiTConfig) {
        self.modelSize = config.textModelSize
        self.numHeads = config.textNumHeads
        self.headDim = config.textModelSize / config.textNumHeads

        self._embedding.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize, dimensions: config.textModelSize
        )
        self._layers.wrappedValue = (0..<config.textNumLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: config.textModelSize,
                numHeads: config.textNumHeads,
                intermediateSize: config.textIntermediateSize
            )
        }
    }

    func callAsFunction(_ tokens: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let seqLen = tokens.dim(1)
        let bsz = tokens.dim(0)
        var h = embedding(tokens)

        let (cos, sin) = echoPrecomputeFreqsCis(dim: headDim, end: seqLen)

        // Non-causal: key mask from padding only
        // Python: mask is [B, seqLen] boolean, broadcast to [B, seqLen, seqLen], then additive
        let attnMask: MLXArray?
        if let m = mask {
            // [B, seqLen] -> broadcast to [B, seqLen, seqLen]
            let keyMask = MLX.broadcast(
                m.reshaped([bsz, 1, seqLen]).asType(Bool.self),
                to: [bsz, seqLen, seqLen]
            )
            attnMask = echoBoolToAdditiveMask(keyMask)  // [B, 1, seqLen, seqLen]
        } else {
            attnMask = nil
        }

        for layer in layers {
            h = layer(h, cosFreqs: cos, sinFreqs: sin, mask: attnMask)
        }
        return h
    }
}

// MARK: - Speaker Encoder

class EchoSpeakerEncoder: Module {
    let modelSize: Int
    let numHeads: Int
    let headDim: Int
    let patchSize: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "blocks") var layers: [EchoEncoderTransformerBlock]

    init(config: EchoDiTConfig) {
        self.modelSize = config.speakerModelSize
        self.numHeads = config.speakerNumHeads
        self.headDim = config.speakerModelSize / config.speakerNumHeads
        self.patchSize = config.speakerPatchSize

        // Input projection from patched latent (Python: bias=True)
        self._inProj.wrappedValue = Linear(
            config.latentSize * config.speakerPatchSize,
            config.speakerModelSize
        )
        self._layers.wrappedValue = (0..<config.speakerNumLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: config.speakerModelSize,
                numHeads: config.speakerNumHeads,
                intermediateSize: config.speakerIntermediateSize
            )
        }
    }

    /// Python SpeakerEncoder.__call__ takes only latent, no mask. Uses causal attention.
    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let bsz = latent.dim(0)
        let seqLen = latent.dim(1)
        let latentSize = latent.dim(2)

        // Truncate to multiple of patchSize
        let seqLenPatched = (seqLen / patchSize) * patchSize
        let truncatedLatent = latent[0..., ..<seqLenPatched, 0...]

        // Patch: [B, T, latentSize] -> [B, T/patchSize, latentSize * patchSize]
        let patchedLen = seqLenPatched / patchSize
        let patched = truncatedLatent.reshaped([bsz, patchedLen, latentSize * patchSize])

        // Project and scale
        var h = inProj(patched) / 6.0

        let (cos, sin) = echoPrecomputeFreqsCis(dim: headDim, end: patchedLen)

        // Python SpeakerEncoder passes mask=None to blocks, but blocks use is_causal=True
        // So the SelfAttention creates causal mask internally
        // We pass a causal mask via the attn mask mechanism
        let causalBool = echoMakeCausalMaskBool(seqLen: patchedLen)
        let attnMask = echoBoolToAdditiveMask(
            causalBool.expandedDimensions(axis: 0).asType(Bool.self)
        )  // [1, 1, seqLen, seqLen]

        for layer in layers {
            h = layer(h, cosFreqs: cos, sinFreqs: sin, mask: attnMask)
        }
        return h
    }
}

// MARK: - EchoDiT (Main Model)

public class EchoDiT: Module {
    let config: EchoDiTConfig

    @ModuleInfo(key: "text_encoder") var textEncoder: EchoTextEncoder
    @ModuleInfo(key: "speaker_encoder") var speakerEncoder: EchoSpeakerEncoder
    @ModuleInfo(key: "text_norm") var textNorm: EchoRMSNorm
    @ModuleInfo(key: "speaker_norm") var speakerNorm: EchoRMSNorm

    // Optional latent encoder for blockwise generation
    @ModuleInfo(key: "latent_encoder") var latentEncoder: EchoSpeakerEncoder?
    @ModuleInfo(key: "latent_norm") var latentNorm: EchoRMSNorm?

    // Condition module: timestep -> 3 * modelSize (for AdaLN shift/scale/gate)
    @ModuleInfo(key: "cond_module") var condModule: EchoSequential

    // Input/output projections
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_norm") var outNorm: EchoRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    // Main transformer blocks (Python: self.blocks)
    @ModuleInfo(key: "blocks") var layers: [EchoTransformerBlock]

    public init(_ config: EchoDiTConfig, hasLatentAttention: Bool = false) {
        self.config = config

        self._textEncoder.wrappedValue = EchoTextEncoder(config: config)
        self._speakerEncoder.wrappedValue = EchoSpeakerEncoder(config: config)
        self._textNorm.wrappedValue = EchoRMSNorm(dim: config.textModelSize)
        self._speakerNorm.wrappedValue = EchoRMSNorm(dim: config.speakerModelSize)

        if hasLatentAttention {
            self._latentEncoder.wrappedValue = EchoSpeakerEncoder(config: config)
            self._latentNorm.wrappedValue = EchoRMSNorm(dim: config.speakerModelSize)
        } else {
            self._latentEncoder.wrappedValue = nil
            self._latentNorm.wrappedValue = nil
        }

        // Condition module: Sequential(Linear, SiLU, Linear, SiLU, Linear)
        // Python: all bias=False
        self._condModule.wrappedValue = EchoSequential(layers: [
            Linear(config.timestepEmbedSize, config.modelSize, bias: false),
            SiLUModule(),
            Linear(config.modelSize, config.modelSize, bias: false),
            SiLUModule(),
            Linear(config.modelSize, config.modelSize * 3, bias: false),
        ])

        // Python: in_proj and out_proj have bias=True
        self._inProj.wrappedValue = Linear(config.latentSize, config.modelSize)
        self._outNorm.wrappedValue = EchoRMSNorm(dim: config.modelSize)
        self._outProj.wrappedValue = Linear(config.modelSize, config.latentSize)

        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            EchoTransformerBlock(config: config, hasLatentAttention: hasLatentAttention)
        }
    }

    /// Encode text and build per-layer KV caches.
    public func getKVCacheText(_ tokens: MLXArray, mask: MLXArray? = nil) -> EchoKVCache {
        var textState = textEncoder(tokens, mask: mask)
        textState = textNorm(textState)
        return layers.map { layer in
            layer.attention.getKVCacheText(textState)
        }
    }

    /// Encode speaker latent and build per-layer KV caches.
    /// Python: get_kv_cache_speaker(speaker_latent) - no mask parameter.
    public func getKVCacheSpeaker(_ latent: MLXArray) -> EchoKVCache {
        var speakerState = speakerEncoder(latent)
        speakerState = speakerNorm(speakerState)
        return layers.map { layer in
            layer.attention.getKVCacheSpeaker(speakerState)
        }
    }

    /// Encode prefix latent (from previous blocks) and build per-layer KV caches.
    /// Used in blockwise generation for temporal coherence between blocks.
    /// Python: get_kv_cache_latent(prefix_latent)
    public func getKVCacheLatent(_ prefixLatent: MLXArray) -> EchoKVCache {
        guard let latentEncoder, let latentNorm else {
            fatalError("Latent prefix modules not loaded. Use delete_blockwise_modules=false.")
        }

        let batchSize = prefixLatent.dim(0)

        // Empty prefix → return empty KV caches
        if prefixLatent.dim(1) == 0 {
            return layers.map { layer in
                (
                    MLXArray.zeros([batchSize, 0, layer.attention.numHeads, layer.attention.headDim]),
                    MLXArray.zeros([batchSize, 0, layer.attention.numHeads, layer.attention.headDim])
                )
            }
        }

        // Encode prefix through latent encoder + norm
        var latentState = latentEncoder(prefixLatent)
        latentState = latentNorm(latentState)

        // Compute RoPE at latent positions (spaced by speaker_patch_size)
        // Python: positions = mx.arange(seq_len) * self.speaker_patch_size
        let seqLen = latentState.dim(1)
        let headDim = config.modelSize / config.numHeads
        let maxPos = seqLen * config.speakerPatchSize
        let (cosAll, sinAll) = echoPrecomputeFreqsCis(dim: headDim, end: maxPos)

        // Index at positions: [0, patchSize, 2*patchSize, ...]
        let positions = (0..<seqLen).map { Int32($0 * config.speakerPatchSize) }
        let posArray = MLXArray(positions)
        let cosLatent = cosAll.take(posArray, axis: 0)
        let sinLatent = sinAll.take(posArray, axis: 0)

        return layers.map { layer in
            layer.attention.getKVCacheLatent(latentState, cosFreqs: cosLatent, sinFreqs: sinLatent)
        }
    }

    /// Forward pass through the DiT.
    public func callAsFunction(
        _ x: MLXArray,
        timestep: MLXArray,
        textKVCache: EchoKVCache,
        speakerKVCache: EchoKVCache,
        latentKVCache: EchoKVCache? = nil,
        textMask: MLXArray? = nil,
        speakerMask: MLXArray? = nil,
        startPos: Int = 0
    ) -> MLXArray {
        let seqLen = x.dim(1)
        let maxPos = startPos + seqLen

        // Timestep conditioning
        let tEmbed = echoGetTimestepEmbedding(timestep, embedSize: config.timestepEmbedSize)
        let condEmbed = condModule(tEmbed).expandedDimensions(axis: 1)  // [B, 1, modelSize*3]

        // Project input
        var h = inProj(x)

        // Precompute RoPE for FULL headDim (Python: precompute_freqs_cis(self.head_dim, max_pos))
        let headDim = config.modelSize / config.numHeads
        let (cos, sin) = echoPrecomputeFreqsCis(dim: headDim, end: maxPos)

        // Downsample speaker mask by patchSize (Python: speaker_mask[..., ::self.speaker_patch_size])
        var dsSpeakerMask = speakerMask
        if let sm = speakerMask {
            let patchSize = config.speakerPatchSize
            dsSpeakerMask = sm[0..., stride(from: 0, to: sm.dim(1), by: patchSize)]
        }

        // Run through transformer blocks
        for (i, layer) in layers.enumerated() {
            let textKV = textKVCache[i]
            let speakerKV = speakerKVCache[i]
            let latentKV = latentKVCache?[i]

            h = layer(
                h,
                condEmbed: condEmbed,
                cosFreqs: cos, sinFreqs: sin,
                startPos: startPos,
                textKV: textKV, speakerKV: speakerKV,
                latentKV: latentKV,
                textMask: textMask, speakerMask: dsSpeakerMask
            )
        }

        // Output projection
        h = outNorm(h)
        h = outProj(h).asType(.float32)
        return h
    }
}

// MARK: - Sequential Wrapper

/// Simple sequential module for the condition module.
class EchoSequential: Module, UnaryLayer {
    @ModuleInfo var layers: [Module]

    init(layers: [Module]) {
        self._layers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let unary = layer as? UnaryLayer {
                h = unary(h)
            }
        }
        return h
    }
}

/// SiLU activation as a module.
class SiLUModule: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(x)
    }
}
