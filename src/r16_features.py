"""Analysis log R16: evidence features per token, the reporting partition of the unresolved class into subtypes, and
the evidence-fusion classifier. Shared by src/r16_benchmark.py (simulated students) and src/r16_flora.py (FLoRA).

Nothing here changes the attribution. The subtypes partition the tokens that the attribution leaves in X by the evidence
the logs hold about their entry; they are a report, not further classes of pi. The fusion classifier is a separate
method that reads the same evidence and returns class probabilities.
Written after R15; see ANALYSIS_LOG_R16.md."""
from difflib import SequenceMatcher
import math
import numpy as np

CLASSES = ['C', 'E', 'S', 'T', 'X']
XSUB = ['X-log-gap', 'X-reentered', 'X-with-C', 'X-near-own-reply', 'X-cohort-reply', 'X-first-snapshot', 'X-no-evidence']
WIN, STRIDE = 20, 10


def windows(tk):
    if len(tk) <= WIN: return [(0, len(tk))] if tk else []
    st_ = list(range(0, len(tk) - WIN + 1, STRIDE))
    if st_[-1] != len(tk) - WIN: st_.append(len(tk) - WIN)
    return [(s, s + WIN) for s in st_]


class Encoder:
    """sentence embeddings of token windows, cached by text"""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import os, pickle
        self.m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); self.cache = {}
        self.path = os.environ.get('EMB_CACHE')   # optional on-disk cache of window embeddings
        if self.path and os.path.exists(self.path): self.cache = pickle.load(open(self.path, 'rb'))

    def save(self):
        import pickle
        if self.path: pickle.dump(self.cache, open(self.path, 'wb'))

    def __call__(self, txts):
        if not txts: return np.zeros((0, 384), dtype=np.float32)
        new = [t for t in dict.fromkeys(txts) if t not in self.cache]
        if new:
            for t, v in zip(new, self.m.encode(new, batch_size=256, normalize_embeddings=True, show_progress_bar=False)):
                self.cache[t] = v.astype(np.float32)
        return np.stack([self.cache[t] for t in txts])


def semantic(A, replies, turns, tokens, enc):
    """per token: max cosine of a covering window of A with a reply window (s_any), the same with the direction check of
    R15 B-21 (s_dir), and the time of the best reply window"""
    wa = windows(A)
    rw, rt, pw, pt = [], [], [], []
    for t, x in replies:
        tk = tokens(x)
        for a, b in windows(tk): rw.append(' '.join(tk[a:b])); rt.append(t)
    for t, x in turns:
        tk = tokens(x)
        for a, b in windows(tk): pw.append(' '.join(tk[a:b])); pt.append(t)
    s_any = np.zeros(len(A)); s_dir = np.zeros(len(A))
    if not wa or not rw: return s_any, s_dir
    EA, ER = enc([' '.join(A[a:b]) for a, b in wa]), enc(rw); EP = enc(pw)
    SR = EA @ ER.T; SP = EA @ EP.T if len(EP) else np.zeros((len(EA), 0)); rt_, pt_ = np.array(rt), np.array(pt)
    for w_, (a, b) in enumerate(wa):
        j = int(SR[w_].argmax()); best = float(SR[w_, j]); s_any[a:b] = np.maximum(s_any[a:b], best)
        earlier = SP[w_][pt_ < rt_[j]] if len(pt_) else np.zeros(0)
        d_ = 0.0 if (len(earlier) and earlier.max() >= best) else best
        s_dir[a:b] = np.maximum(s_dir[a:b], d_)
    return s_any, s_dir


def cohort_max(A, enc, pool_emb, own_mask=None):
    """per token: max cosine of a covering window of A with a pool of reply windows given to other writers"""
    wa = windows(A); out = np.zeros(len(A))
    if not wa or pool_emb is None or not len(pool_emb): return out
    EA = enc([' '.join(A[a:b]) for a, b in wa]); S = EA @ pool_emb.T
    if own_mask is not None: S[:, own_mask] = -1
    best = S.max(axis=1)
    for w_, (a, b) in enumerate(wa): out[a:b] = np.maximum(out[a:b], best[w_])
    return out


def features(st, o, n, tau, tokens, ngrams, fuzzyC, s_any, s_dir, cohort_gram, s_cohort):
    """evidence features per token of the final artefact.
    st: replies, turns, snaps (times in seconds); o: output of attribute(..., keep=True, entry_rule='token');
    fuzzyC: token marks of tolerant matching; cohort_gram(g) -> True if g occurs in a reply given to another writer."""
    A, cls, lab, tb, tp, te = o['A'], o['cls'], o['lab'], o['tb'], o['tp'], o['te']
    last, tl, sid, seg = o['last'], o['tl'], o['sid'], o['seg']
    m = len(A); G = ngrams(A, n)
    io = [None] * m; sd = [None] * m; sg = [None] * m
    for op, i1, i2, j1, j2 in SequenceMatcher(None, A, last, autojunk=False).get_opcodes():
        if op == 'equal':
            for q in range(i2 - i1): io[i1 + q] = tl[j1 + q]; sd[i1 + q] = sid[j1 + q]; sg[i1 + q] = seg[j1 + q]
    snapt = [t for t, _ in st['snaps']]
    first_nonempty = next((k for k, (t, x) in enumerate(st['snaps']) if tokens(x)), 0)
    cnt = {c: np.zeros(m) for c in CLASSES}; inrep = np.zeros(m); inturn = np.zeros(m); incoh = np.zeros(m)
    gap = np.full(m, np.nan); reent = np.zeros(m)
    for i, g in enumerate(G):
        lg = lab[g]; rep = g in tb; tur = g in tp; coh = cohort_gram(g)
        gp = (te[g] - tb[g]) if (rep and g in te) else np.nan
        for j in range(i, i + n):
            cnt[lg][j] += 1; inrep[j] += rep; inturn[j] += tur; incoh[j] += coh
            if not np.isnan(gp) and (np.isnan(gap[j]) or abs(gp) < abs(gap[j])): gap[j] = gp
    # re-entered: a covering n-gram was in the editor before the snapshot that brought the token in
    for i, g in enumerate(G):
        if g in te:
            for j in range(i, i + n):
                if sd[j] is not None and te[g] < snapt[sd[j]] - 1e-9: reent[j] = 1
    segC = {}
    for j in range(m):
        if sg[j] is not None and cls[j] == 'C': segC[sg[j]] = True
    segsize = {}
    for j in range(m):
        if sg[j] is not None: segsize[sg[j]] = segsize.get(sg[j], 0) + 1
    rows = []
    for j in range(m):
        e = io[j]
        r = [cls[j] == c for c in CLASSES] + [cnt[c][j] for c in CLASSES]
        r += [inrep[j], inturn[j], incoh[j], -1.0 if e is None else math.log1p(e), float(e is None),
              float(sd[j] == first_nonempty) if sd[j] is not None else 0.0, float(segC.get(sg[j], False)),
              math.log1p(segsize.get(sg[j], 0)), float(fuzzyC[j]), float(s_any[j]), float(s_dir[j]), float(s_cohort[j]),
              0.0 if np.isnan(gap[j]) else math.copysign(math.log1p(abs(gap[j])), gap[j]), float(np.isnan(gap[j])), reent[j]]
        rows.append(r)
    X = np.array(rows, dtype=np.float32)
    aux = dict(io=io, sid=sd, seg=sg, first_nonempty=first_nonempty, reent=reent, segC=[bool(segC.get(s, False)) for s in sg],
               incoh=incoh, s_dir=np.asarray(s_dir), s_cohort=np.asarray(s_cohort))
    return X, aux


FEATURES = ([f'rule_{c}' for c in CLASSES] + [f'ngrams_{c}' for c in CLASSES] +
            ['ngrams_in_reply', 'ngrams_in_turn', 'ngrams_in_cohort_reply', 'log_entry_size', 'entry_unknown',
             'first_snapshot', 'segment_has_C', 'log_segment_size', 'tolerant_C', 'sem_reply_any', 'sem_reply_dir',
             'sem_cohort', 'signed_log_gap_editor_minus_reply', 'no_reply_gap', 'reentered'])


def x_subtypes(cls, aux, theta_own, theta_cohort):
    """reporting partition of X by the first condition that holds, in the order of XSUB"""
    out = []
    for j, c in enumerate(cls):
        if c != 'X': out.append(None); continue
        if aux['io'][j] is None: out.append('X-log-gap')
        elif aux['reent'][j]: out.append('X-reentered')
        elif aux['segC'][j]: out.append('X-with-C')
        elif aux['s_dir'][j] >= theta_own: out.append('X-near-own-reply')
        elif aux['incoh'][j] > 0 or aux['s_cohort'][j] >= theta_cohort: out.append('X-cohort-reply')
        elif aux['sid'][j] == aux['first_nonempty']: out.append('X-first-snapshot')
        else: out.append('X-no-evidence')
    return out


def ece(prob, y, bins=15):
    """expected calibration error of the top label"""
    conf = prob.max(1); pred = prob.argmax(1); acc = (pred == y).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins + 1); rel = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (conf > a) & (conf <= b)
        if k.any():
            e += k.mean() * abs(acc[k].mean() - conf[k].mean()); rel.append((float(a), float(b), int(k.sum()), float(conf[k].mean()), float(acc[k].mean())))
    return float(e), rel


def ece_binary(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; rel = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if k.any():
            e += k.mean() * abs(y[k].mean() - p[k].mean()); rel.append((float(a), float(b), int(k.sum()), float(p[k].mean()), float(y[k].mean())))
    return float(e), rel
