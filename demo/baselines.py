"""Overlap baselines used for comparison with the five-class attribution (standard library only).

winnow_mark   fingerprint matching in the manner of MOSS (Schleimer, Wilkerson & Aiken, 2003): hashes of token
              k-grams, one minimum per window of w hashes; a k-gram of the target whose fingerprint is also a
              fingerprint of the source marks its k tokens.
tile_mark     greedy string tiling in the manner of JPlag (Prechelt, Malpohl & Philippsen, 2002): maximal common
              token runs of at least mml tokens are tiled longest first, each source and target token used once.
overlap_mark  any target token covered by a source n-gram (naive overlap).
All functions take token lists and return a list of booleans, one per target token.
"""
import zlib


def _h(gram):
    return zlib.crc32(" ".join(gram).encode("utf8"))


def fingerprints(toks, k=8, w=4):
    """(position, hash) pairs selected by winnowing: the rightmost minimum of each window of w k-gram hashes."""
    hs = [_h(toks[i:i + k]) for i in range(len(toks) - k + 1)]
    if not hs:
        return []
    if len(hs) <= w:
        j = min(range(len(hs)), key=lambda i: (hs[i], -i))
        return [(j, hs[j])]
    out, last = [], -1
    for s in range(len(hs) - w + 1):
        win = range(s, s + w)
        j = min(win, key=lambda i: (hs[i], -i))
        if j != last:
            out.append((j, hs[j]))
            last = j
    return out


def winnow_mark(target, sources, k=8, w=4):
    src = {h for s in sources for _, h in fingerprints(s, k, w)}
    m = [False] * len(target)
    for j, h in fingerprints(target, k, w):
        if h in src:
            for i in range(j, min(j + k, len(target))):
                m[i] = True
    return m


def overlap_mark(target, sources, n=8):
    src = {tuple(s[i:i + n]) for s in sources for i in range(len(s) - n + 1)}
    m = [False] * len(target)
    for i in range(len(target) - n + 1):
        if tuple(target[i:i + n]) in src:
            for j in range(i, i + n):
                m[j] = True
    return m


def tile_mark(target, sources, mml=8):
    """Greedy string tiling of target against the concatenation of sources (a separator keeps sources apart)."""
    src = []
    for s in sources:
        src += list(s) + [None]
    mt, ms = [False] * len(target), [False] * len(src)
    idx = {}
    for j in range(len(src) - mml + 1):
        g = tuple(src[j:j + mml])
        if None not in g:
            idx.setdefault(g, []).append(j)
    while True:
        best, matches = mml - 1, []
        for i in range(len(target) - mml + 1):
            if mt[i]:
                continue
            g = tuple(target[i:i + mml])
            for j in idx.get(g, ()):
                if ms[j]:
                    continue
                L = 0
                while (i + L < len(target) and j + L < len(src) and not mt[i + L] and not ms[j + L]
                       and src[j + L] is not None and target[i + L] == src[j + L]):
                    L += 1
                if L > best:
                    best, matches = L, [(i, j)]
                elif L == best:
                    matches.append((i, j))
        if best < mml:
            break
        for i, j in matches:
            if any(mt[i:i + best]) or any(ms[j:j + best]):
                continue
            for q in range(best):
                mt[i + q] = True
                ms[j + q] = True
    return mt
