import Foundation

/// Word-Error-Rate (WER) and Character-Error-Rate (CER) calculation for STT
/// benchmark scoring.
///
/// - Standard WER  = (Insertions + Deletions + Substitutions) / Reference word count
/// - Semantic WER  = same formula on normalized text (punctuation stripped,
///                   contractions expanded, filler words removed)
/// - CER           = same edit-distance algorithm at character granularity
///
/// Ported from Sam (`Sam/Tests/Benchmarks/WERCalculator.swift`) so this
/// scoring code lives next to the canonical STT benchmark suite.
enum WERCalculator {

    /// Detailed result including per-operation breakdown.
    struct ErrorRateResult {
        let rate: Double
        let insertions: Int
        let deletions: Int
        let substitutions: Int
        let referenceUnitCount: Int
        let hypothesisUnitCount: Int
    }

    // MARK: - WER

    /// Standard WER: lowercase, whitespace-tokenized.
    static func wer(reference: String, hypothesis: String) -> ErrorRateResult {
        let refWords = tokenize(reference)
        let hypWords = tokenize(hypothesis)
        return editDistance(reference: refWords, hypothesis: hypWords)
    }

    /// Semantic WER: ignores punctuation, expands contractions, drops filler
    /// words. Catches meaning-altering errors more reliably than raw WER.
    static func semanticWER(reference: String, hypothesis: String) -> ErrorRateResult {
        let refWords = tokenizeSemantic(reference)
        let hypWords = tokenizeSemantic(hypothesis)
        return editDistance(reference: refWords, hypothesis: hypWords)
    }

    // MARK: - CER

    /// Character Error Rate: edit distance over individual characters of the
    /// lowercased input. Useful as a complementary metric to WER —
    /// near-misses on a single phoneme show up as high WER but low CER.
    static func cer(reference: String, hypothesis: String) -> ErrorRateResult {
        let refChars = reference.lowercased().map(String.init)
        let hypChars = hypothesis.lowercased().map(String.init)
        return editDistance(reference: refChars, hypothesis: hypChars)
    }

    // MARK: - Tokenization

    private static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .split(separator: " ", omittingEmptySubsequences: true)
            .map(String.init)
    }

    private static func tokenizeSemantic(_ text: String) -> [String] {
        var normalized = text.lowercased()
        for (contraction, expanded) in contractionMap {
            normalized = normalized.replacingOccurrences(of: contraction, with: expanded)
        }
        normalized = normalized.unicodeScalars
            .filter { CharacterSet.alphanumerics.contains($0) || CharacterSet.whitespaces.contains($0) }
            .map { String($0) }
            .joined()
        return normalized.split(separator: " ", omittingEmptySubsequences: true)
            .map(String.init)
            .filter { !fillerWords.contains($0) }
    }

    // MARK: - Edit distance (Wagner–Fischer)

    private static func editDistance(reference: [String], hypothesis: [String]) -> ErrorRateResult {
        let n = reference.count
        let m = hypothesis.count

        guard n > 0 else {
            return ErrorRateResult(
                rate: m > 0 ? 1.0 : 0.0,
                insertions: m, deletions: 0, substitutions: 0,
                referenceUnitCount: 0, hypothesisUnitCount: m
            )
        }
        guard m > 0 else {
            return ErrorRateResult(
                rate: 1.0,
                insertions: 0, deletions: n, substitutions: 0,
                referenceUnitCount: n, hypothesisUnitCount: 0
            )
        }

        struct Cell {
            var cost: Int
            var insertions: Int
            var deletions: Int
            var substitutions: Int
        }

        var dp = Array(
            repeating: Array(repeating: Cell(cost: 0, insertions: 0, deletions: 0, substitutions: 0), count: m + 1),
            count: n + 1
        )

        for i in 1...n {
            dp[i][0] = Cell(cost: i, insertions: 0, deletions: i, substitutions: 0)
        }
        for j in 1...m {
            dp[0][j] = Cell(cost: j, insertions: j, deletions: 0, substitutions: 0)
        }

        for i in 1...n {
            for j in 1...m {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    let sub = dp[i - 1][j - 1]
                    let del = dp[i - 1][j]
                    let ins = dp[i][j - 1]

                    if sub.cost <= del.cost && sub.cost <= ins.cost {
                        dp[i][j] = Cell(
                            cost: sub.cost + 1,
                            insertions: sub.insertions,
                            deletions: sub.deletions,
                            substitutions: sub.substitutions + 1
                        )
                    } else if del.cost <= ins.cost {
                        dp[i][j] = Cell(
                            cost: del.cost + 1,
                            insertions: del.insertions,
                            deletions: del.deletions + 1,
                            substitutions: del.substitutions
                        )
                    } else {
                        dp[i][j] = Cell(
                            cost: ins.cost + 1,
                            insertions: ins.insertions + 1,
                            deletions: ins.deletions,
                            substitutions: ins.substitutions
                        )
                    }
                }
            }
        }

        let final = dp[n][m]
        // Cap at 1.0 — uncapped WER can exceed 1.0 when insertions outnumber
        // reference words, which is more confusing than informative on a
        // benchmark report.
        let rate = min(Double(final.cost) / Double(n), 1.0)

        return ErrorRateResult(
            rate: rate,
            insertions: final.insertions,
            deletions: final.deletions,
            substitutions: final.substitutions,
            referenceUnitCount: n,
            hypothesisUnitCount: m
        )
    }

    // MARK: - Constants

    private static let fillerWords: Set<String> = [
        "uh", "um", "uhm", "hmm", "hm", "ah", "oh", "er", "like",
        "you know", "i mean", "well", "so", "basically", "actually",
        "right", "okay", "ok",
    ]

    private static let contractionMap: [(String, String)] = [
        ("won't", "will not"),
        ("can't", "cannot"),
        ("couldn't", "could not"),
        ("shouldn't", "should not"),
        ("wouldn't", "would not"),
        ("didn't", "did not"),
        ("doesn't", "does not"),
        ("don't", "do not"),
        ("isn't", "is not"),
        ("aren't", "are not"),
        ("wasn't", "was not"),
        ("weren't", "were not"),
        ("hasn't", "has not"),
        ("haven't", "have not"),
        ("hadn't", "had not"),
        ("i'm", "i am"),
        ("you're", "you are"),
        ("we're", "we are"),
        ("they're", "they are"),
        ("he's", "he is"),
        ("she's", "she is"),
        ("it's", "it is"),
        ("that's", "that is"),
        ("there's", "there is"),
        ("who's", "who is"),
        ("what's", "what is"),
        ("i've", "i have"),
        ("you've", "you have"),
        ("we've", "we have"),
        ("they've", "they have"),
        ("i'll", "i will"),
        ("you'll", "you will"),
        ("we'll", "we will"),
        ("they'll", "they will"),
        ("i'd", "i would"),
        ("you'd", "you would"),
        ("we'd", "we would"),
        ("they'd", "they would"),
        ("let's", "let us"),
        ("gonna", "going to"),
        ("wanna", "want to"),
        ("gotta", "got to"),
    ]
}
