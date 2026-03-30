# EVOLVE-BLOCK-START
def similarity(a: str, b: str) -> float:
    """
    Return a similarity score between 0.0 (unrelated) and 1.0 (identical)
    for two input strings.

    This should capture not just character-level similarity but also
    meaning — paraphrases should score high, negations should score low,
    and typos should be forgiven.

    Only use the Python standard library (no external packages).
    """
    # Baseline: normalized Levenshtein distance
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp

    max_len = max(m, n)
    return 1.0 - dp[n] / max_len
# EVOLVE-BLOCK-END
