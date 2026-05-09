import Foundation

/// Central registry of audio model families with their HuggingFace repo IDs
/// per quantization variant. Mirrors `mlx-swift-lm/Tests/Benchmarks/Utils/ModelRegistry.swift`
/// in shape so a benchmark report's "Config" column reads the same across
/// repos: `<short-name> / <quant>`.
///
/// Adding a new model is intentionally just data: drop a `ModelFamily` entry
/// into the relevant pipeline list. The runners look it up by short-name
/// (the value of `MLX_AUDIO_BENCH_MODEL`) and pick the variant matching
/// `MLX_AUDIO_BENCH_QUANT` (defaulting to the first variant when unset).
///
/// Variants with empty `repoId` strings are placeholders for "we want to
/// bench this once a release is published" — the runner emits a `[BENCH]
/// skipped` line and moves on. As mlx-community publishes more variants,
/// these slots get filled in without changing runner code.
///
/// Repo IDs in this file are verified against `Tests/MLXAudioSmokeTests.swift`
/// where possible. New entries that aren't in smoke tests are marked with
/// `notes: "unverified repoId"` so a reviewer can confirm before relying on
/// them.
enum ModelRegistry {

    enum Pipeline: String {
        case stt
        case tts
        case codec
        case vad
        case lid
        case sts
    }

    struct ModelVariant {
        /// Short identifier — `bf16`, `4bit`, `8bit`, `fp16`, `6bit`, etc.
        let quantization: String
        /// Full HuggingFace repo ID. Empty string means "not available yet"
        /// — the runner will skip and emit a `[BENCH] skipped` line.
        let repoId: String
        /// Optional sub-file when the repo packages multiple checkpoints
        /// (e.g., Mimi's tokenizer .safetensors). Most models leave this nil.
        let filename: String?

        init(quantization: String, repoId: String, filename: String? = nil) {
            self.quantization = quantization
            self.repoId = repoId
            self.filename = filename
        }
    }

    struct ModelFamily {
        let pipeline: Pipeline
        /// Display name — appears in report headings.
        let name: String
        /// CLI short-name — must be unique within a pipeline.
        let shortName: String
        /// Variants in preference order. The runner picks the first that
        /// matches the requested quant; fallback is variant[0].
        let variants: [ModelVariant]
        /// Optional notes — emitted into the report's parameter table for
        /// reviewers (e.g. "TDT decoder", "non-streaming only").
        let notes: String?
        /// True if the model needs a reference audio clip for inference
        /// (voice cloning, e.g. Chatterbox, Marvis, FishSpeech). The TTS
        /// runner will load `MLX_AUDIO_BENCH_REF_AUDIO` (with a fallback
        /// to the bundled reference-outputs directory) when this is set.
        let requiresReferenceAudio: Bool

        init(
            pipeline: Pipeline,
            name: String,
            shortName: String,
            variants: [ModelVariant],
            notes: String? = nil,
            requiresReferenceAudio: Bool = false
        ) {
            self.pipeline = pipeline
            self.name = name
            self.shortName = shortName
            self.variants = variants
            self.notes = notes
            self.requiresReferenceAudio = requiresReferenceAudio
        }
    }

    // MARK: - Lookup

    static func family(named shortName: String, pipeline: Pipeline) -> ModelFamily? {
        all.first { $0.pipeline == pipeline && $0.shortName == shortName }
    }

    static func families(in pipeline: Pipeline) -> [ModelFamily] {
        all.filter { $0.pipeline == pipeline }
    }

    /// Wrap an arbitrary HuggingFace repo into a one-shot family. Useful for
    /// benchmarking models that aren't in the built-in registry yet.
    static func customFamily(repoId: String, pipeline: Pipeline, shortName: String? = nil) -> ModelFamily {
        let derived = shortName ?? repoId.split(separator: "/").last.map(String.init) ?? repoId
        return ModelFamily(
            pipeline: pipeline,
            name: repoId,
            shortName: derived,
            variants: [ModelVariant(quantization: "as-published", repoId: repoId)],
            notes: "custom repo"
        )
    }

    // MARK: - Built-in registry

    /// All registered families.
    static let all: [ModelFamily] = sttFamilies + ttsFamilies + codecFamilies + vadFamilies + lidFamilies + stsFamilies

    // MARK: STT (verified against smoke tests)

    static let sttFamilies: [ModelFamily] = [
        ModelFamily(
            pipeline: .stt,
            name: "Parakeet TDT 0.6B v2",
            shortName: "parakeet-tdt-0.6b-v2",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/parakeet-tdt-0.6b-v2"),
            ],
            notes: "TDT decoder; English-only; supports batched inference"
        ),
        ModelFamily(
            pipeline: .stt,
            name: "Parakeet TDT 0.6B v3",
            shortName: "parakeet-tdt-0.6b-v3",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/parakeet-tdt-0.6b-v3"),
            ],
            notes: "TDT decoder; multilingual; supports batched inference"
        ),
        ModelFamily(
            pipeline: .stt,
            name: "Qwen3 ASR 0.6B",
            shortName: "qwen3-asr-0.6b",
            variants: [
                ModelVariant(quantization: "4bit", repoId: "mlx-community/Qwen3-ASR-0.6B-4bit"),
                ModelVariant(quantization: "bf16", repoId: "mlx-community/Qwen3-ASR-0.6B-bf16"),
            ],
            notes: "LLM-style decoder; multilingual"
        ),
        ModelFamily(
            pipeline: .stt,
            name: "Qwen3 ASR 1.7B",
            shortName: "qwen3-asr-1.7b",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/Qwen3-ASR-1.7B-bf16"),
            ],
            notes: "Larger Qwen3-ASR; multilingual"
        ),
        ModelFamily(
            pipeline: .stt,
            name: "GLM ASR Nano",
            shortName: "glm-asr-nano",
            variants: [
                ModelVariant(quantization: "4bit", repoId: "mlx-community/GLM-ASR-Nano-2512-4bit"),
            ],
            notes: "GLM-based ASR"
        ),
        ModelFamily(
            pipeline: .stt,
            name: "Granite Speech 1B",
            shortName: "granite-speech-1b",
            variants: [
                ModelVariant(quantization: "5bit", repoId: "mlx-community/granite-4.0-1b-speech-5bit"),
            ],
            notes: ""
        ),
    ]

    // MARK: TTS (verified against smoke tests)

    static let ttsFamilies: [ModelFamily] = [
        ModelFamily(
            pipeline: .tts,
            name: "Kokoro 82M",
            shortName: "kokoro-82m",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/Kokoro-82M-bf16"),
            ],
            notes: "StyleTTS2 family; default voice af_heart; 24kHz output"
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Kitten TTS",
            shortName: "kitten-tts",
            variants: [
                ModelVariant(quantization: "fp16", repoId: "mlx-community/kitten-tts-nano-0.8-fp16"),
                ModelVariant(quantization: "8bit", repoId: "mlx-community/kitten-tts-nano-0.8-8bit"),
                ModelVariant(quantization: "mini-fp16", repoId: "mlx-community/kitten-tts-mini-0.8-fp16"),
            ],
            notes: "StyleTTS2 family, smallest variant; 24kHz output"
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Echo TTS",
            shortName: "echo-tts",
            variants: [
                ModelVariant(quantization: "base", repoId: "mlx-community/echo-tts-base"),
            ],
            notes: "Higher-fidelity than Kokoro at the cost of speed"
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Chatterbox",
            shortName: "chatterbox",
            variants: [
                ModelVariant(quantization: "fp16", repoId: "mlx-community/Chatterbox-TTS-fp16"),
                ModelVariant(quantization: "fp16-turbo", repoId: "mlx-community/chatterbox-turbo-fp16"),
            ],
            notes: "Voice cloning; reference-audio conditioned",
            requiresReferenceAudio: true
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Marvis TTS (CSM/Sesame)",
            shortName: "marvis-tts",
            variants: [
                ModelVariant(quantization: "8bit", repoId: "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"),
                ModelVariant(quantization: "4bit", repoId: "Marvis-AI/marvis-tts-250m-v0.2-MLX-4bit"),
                ModelVariant(quantization: "100m-8bit", repoId: "Marvis-AI/marvis-tts-100m-v0.2-MLX-8bit"),
            ],
            notes: "CSM-based; voice cloning via reference audio",
            requiresReferenceAudio: true
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Pocket TTS",
            shortName: "pocket-tts",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/pocket-tts"),
            ],
            notes: "Smallest, fastest TTS"
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Soprano",
            shortName: "soprano",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/Soprano-1.1-80M-bf16"),
                ModelVariant(quantization: "bf16-80m", repoId: "mlx-community/Soprano-80M-bf16"),
            ],
            notes: ""
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Vyvo TTS (Qwen3 backbone)",
            shortName: "vyvo-tts",
            variants: [
                ModelVariant(quantization: "4bit", repoId: "mlx-community/VyvoTTS-EN-Beta-4bit"),
            ],
            notes: "Qwen3-based TTS"
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Orpheus 3B",
            shortName: "orpheus-3b",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/orpheus-3b-0.1-ft-bf16"),
            ],
            notes: "Llama-backbone TTS"
        ),
        ModelFamily(
            pipeline: .tts,
            name: "Qwen3 TTS 0.6B Base",
            shortName: "qwen3-tts",
            variants: [
                ModelVariant(quantization: "4bit", repoId: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"),
                ModelVariant(quantization: "8bit", repoId: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"),
                ModelVariant(quantization: "bf16", repoId: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"),
            ],
            notes: "Qwen3 audio-token TTS; multilingual"
        ),
        ModelFamily(
            pipeline: .tts,
            name: "FishSpeech S2 Pro",
            shortName: "fish-speech",
            variants: [
                ModelVariant(quantization: "8bit", repoId: "mlx-community/fish-audio-s2-pro-8bit"),
                ModelVariant(quantization: "bf16", repoId: "mlx-community/fish-audio-s2-pro-bf16"),
            ],
            notes: "Voice cloning; reference-audio + reference-text conditioned",
            requiresReferenceAudio: true
        ),
        ModelFamily(
            pipeline: .tts,
            name: "MOSS-TTS Nano 100M",
            shortName: "moss-tts-nano",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/MOSS-TTS-Nano-100M"),
            ],
            notes: "MOSS Nano; voice-clone via reference audio",
            requiresReferenceAudio: true
        ),
        ModelFamily(
            pipeline: .tts,
            name: "MOSS-TTS 8B",
            shortName: "moss-tts",
            variants: [
                ModelVariant(quantization: "8bit", repoId: "mlx-community/MOSS-TTS-8B-8bit"),
            ],
            notes: "MOSS-TTSD 8B; multilingual + multi-speaker; voice-clone via reference audio",
            requiresReferenceAudio: true
        ),
    ]

    // MARK: Codec (verified against smoke tests)

    static let codecFamilies: [ModelFamily] = [
        ModelFamily(
            pipeline: .codec,
            name: "SNAC 24kHz",
            shortName: "snac-24khz",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "mlx-community/snac_24khz"),
            ],
            notes: "3-level codebook; 24kHz output"
        ),
        ModelFamily(
            pipeline: .codec,
            name: "Mimi",
            shortName: "mimi",
            variants: [
                ModelVariant(
                    quantization: "bf16",
                    repoId: "kyutai/moshiko-pytorch-bf16",
                    filename: "tokenizer-e351c8d8-checkpoint125.safetensors"
                ),
            ],
            notes: "Kyutai Moshi tokenizer"
        ),
    ]

    // MARK: VAD / Diarization (verified)

    static let vadFamilies: [ModelFamily] = [
        ModelFamily(
            pipeline: .vad,
            name: "Sortformer Streaming 4spk",
            shortName: "sortformer-streaming-4spk",
            variants: [
                ModelVariant(quantization: "fp16", repoId: "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"),
            ],
            notes: "Speaker diarization; offline + streaming"
        ),
        ModelFamily(
            pipeline: .vad,
            name: "Silero VAD",
            shortName: "silero-vad",
            variants: [
                ModelVariant(quantization: "fp32", repoId: "mlx-community/silero-vad"),
                ModelVariant(quantization: "fp32-v6", repoId: "mlx-community/silero-vad-v6"),
            ],
            notes: "Lightweight voice activity detection; speech-presence only"
        ),
    ]

    // MARK: LID (verified)

    static let lidFamilies: [ModelFamily] = [
        ModelFamily(
            pipeline: .lid,
            name: "MMS-LID-256 (Wav2Vec2)",
            shortName: "mms-lid-256",
            variants: [
                ModelVariant(quantization: "bf16", repoId: "facebook/mms-lid-256"),
            ],
            notes: "256-language coverage"
        ),
    ]

    // MARK: STS (verified)

    static let stsFamilies: [ModelFamily] = [
        ModelFamily(
            pipeline: .sts,
            name: "LFM2.5 Audio 1.5B",
            shortName: "lfm25-audio-1.5b",
            variants: [
                ModelVariant(quantization: "6bit", repoId: "mlx-community/LFM2.5-Audio-1.5B-6bit"),
            ],
            notes: "Speech understanding"
        ),
    ]
}

extension ModelRegistry.ModelFamily {
    /// Resolve the variant matching `quant`, falling back to the first
    /// variant when `quant` is nil or no exact match exists. Emits a
    /// warning to stderr in the fallback path so the operator notices.
    func resolveVariant(_ quant: String?) -> ModelRegistry.ModelVariant {
        if let q = quant, let match = variants.first(where: { $0.quantization == q }) {
            return match
        }
        if quant != nil, !variants.isEmpty {
            FileHandle.standardError.write(Data(
                "[BENCH] warning: no variant '\(quant ?? "")' for '\(shortName)' — using '\(variants[0].quantization)'\n".utf8
            ))
        }
        return variants[0]
    }
}
