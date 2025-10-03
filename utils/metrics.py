# utils/metrics.py
import re, math

def normalize_text(s):
    s = (s or "").lower()
    s = s.replace("\n", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def edit_distance(a, b):
    n = len(a); m = len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
    return dp[n][m]

def cer(ref, hyp):
    r = ref
    h = hyp
    if len(r) == 0:
        return float(len(h) > 0)
    return edit_distance(list(r), list(h)) / max(1, len(r))

def wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    if len(r) == 0:
        return float(len(h) > 0)
    return edit_distance(r, h) / max(1, len(r))

def bleu_simple(ref, hyp):
    r = ref.split(); h = hyp.split()
    if len(h) == 0: return 0.0
    ref_counts = {}
    for w in r: ref_counts[w] = ref_counts.get(w, 0) + 1
    match = 0
    for w in h:
        if ref_counts.get(w, 0) > 0:
            match += 1
            ref_counts[w] -= 1
    prec = match / len(h)
    bp = 1.0
    if len(h) < len(r) and len(h) > 0:
        bp = math.exp(1 - len(r) / len(h))
    return bp * prec

def lcs(a, b):
    n = len(a); m = len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

def rouge_l(ref, hyp):
    a = ref.split(); b = hyp.split()
    if len(a) == 0 or len(b) == 0: return 0.0
    l = lcs(a, b)
    prec = l / len(b)
    rec = l / len(a)
    if prec + rec == 0: return 0.0
    beta = 1.2
    return ((1+beta**2) * prec * rec) / (rec + beta**2 * prec + 1e-12)
