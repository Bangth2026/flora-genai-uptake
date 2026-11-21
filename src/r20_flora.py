"""Analysis log R20, F-44 to F-47 (FLoRA). Written after all earlier results were known; nothing in the attribution
changes. Definitions of src/r9b_flora_token_entry.py (n = 8, tau = 5, delta = 0, peer threshold 5).
F-44 earliest plausible attribution (EPA) on the field logs: (a) n-gram EPA over the two-author revision sequence of
Property 3 (prompts and snapshots by the writer, replies by the chatbot, ties to the chatbot, no shared sources);
(b) the same without prompts (replies and snapshots only); (c) EPA as WikiWho implements it (src/r20_wikiwho.py) on the
raw texts, with every snapshot. Per variant: mean and median share of chatbot text, share of students at or above
25/50/75% with student-bootstrap intervals, rank correlation with U, token agreement with class C.
F-45 the echo class split by whether the token entered the editor after a reply that contained it (an E 8-gram whose
first editor time is not earlier than its first reply time; only prompt-first echo can satisfy this): share of E, of
large-entry E, and the share of students at or above half when C is joined by (i) echo that entered after a reply
containing it, (ii) the large-entry part of (i).
F-46 C 8-grams made mostly of numerals (at least four of eight tokens contain a digit) or of function words (at least
six of eight in scikit-learn's English stop-word list): share of C 8-grams, and share of C tokens covered only by such
8-grams.
F-47 coincidence with replies to similar prompts: for each student, every prompt is matched to the most similar prompt
of another student (TF-IDF cosine over word unigrams and bigrams), and the proposal's coverage by 8-grams of the replies
to those matched prompts is compared with its coverage by the replies of a random other student (as in the chance
baseline of R4); both with and without 8-grams of shared sources; 5 random draws, seed 2201.
Usage: python -I src/r20_flora.py DATA_DIR -> results/r20_flora.json"""
import sys, os, json, time
import numpy as np, pandas as pd
from difflib import SequenceMatcher
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
D = sys.argv[1]
t00 = time.perf_counter()
def log(*a): print(round(time.perf_counter() - t00), *a, flush=True)
src = open(os.path.join(HERE, 'r9b_flora_token_entry.py')).read()
cut = src.index('# ======================= R9b body')
sys.argv = [os.path.join(HERE, 'r9b_flora_token_entry.py'), D]
G = {'__name__': 'r9b_defs', '__file__': os.path.join(HERE, 'r9b_flora_token_entry.py')}
exec(compile(src[:cut], 'r9b_flora_token_entry.py', 'exec'), G)
d, users, P, WG, tok, grams, scan, classify, project, dialogue, H, Cc = (G[k] for k in
    ('d', 'users_all', 'P', 'WG', 'tok', 'grams', 'scan', 'classify', 'project', 'dialogue', 'H', 'Cc'))
from r20_wikiwho import wikiwho_chatbot_marks
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
NB = 2000; rng = np.random.default_rng(2201); N = 8

def size_of_entry(enc, F, Fi):
    """size of the inserted segment that brought each final token in (as src/r19_flora.py, without keystrokes)"""
    prev = np.zeros(0, np.int32); size = np.zeros(0, np.int64)
    for cur in enc:
        L = len(cur)
        if len(prev) == 0:
            size = np.full(L, L, np.int64); prev = cur; continue
        m = min(len(prev), L); ne = np.flatnonzero(prev[:m] != cur[:m]); pre = int(ne[0]) if len(ne) else m
        mm = m - pre; ne2 = np.flatnonzero(prev[::-1][:mm] != cur[::-1][:mm]); suf = int(ne2[0]) if len(ne2) else mm
        a, b = prev[pre:len(prev) - suf], cur[pre:L - suf]
        if len(a) == 0 and len(b) == 0: prev = cur; continue
        ns = np.empty(L, np.int64); ns[:pre] = size[:pre]
        if suf: ns[L - suf:] = size[len(prev) - suf:]
        if len(b):
            ops = [('insert', 0, 0, 0, len(b))] if len(a) == 0 else SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False).get_opcodes()
            for op, i1, i2, j1, j2 in ops:
                if op == 'equal': ns[pre + j1:pre + j2] = size[pre + i1:pre + i2]
                elif op in ('insert', 'replace'): ns[pre + j1:pre + j2] = j2 - j1
        size, prev = ns, cur
    out = np.full(len(F), 10 ** 9, np.int64)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, Fi.tolist(), prev.tolist(), autojunk=False).get_opcodes():
        if op == 'equal': out[i1:i2] = size[j1:j2]
    return out

STOP = set(ENGLISH_STOP_WORDS)
def is_num(g): return sum(any(ch.isdigit() for ch in x) for x in g) >= 4
def is_func(g): return sum(x in STOP for x in g) >= 6

rows = []; agreeC = []; f46 = {'C_ngrams': 0, 'numeric': 0, 'function': 0, 'C_tokens': 0, 'C_tokens_only_numeric_or_function': 0}
for kk, u in enumerate(users):
    g_ = WG.get(u); times = list(g_.t) if g_ is not None else []; vocab = {}
    enc = [np.array([vocab.setdefault(x, len(vocab)) for x in tok(c)], np.int32) for c in (g_.writing_content if g_ is not None else [])]
    F = P[u]; Fi = np.array([vocab.setdefault(x, len(vocab)) for x in F], np.int32)
    tbot, tpr = dialogue(u, N); FG = grams(F, N); FGs = {g: tuple(Fi[i:i + N].tolist()) for i, g in enumerate(FG)}
    sc = scan(enc, times, list(range(len(enc))), N, set(FGs.values()))
    cls = classify(FGs, tbot, tpr, sc, N); lab = project(F, N, cls)
    size = size_of_entry(enc, F, Fi) if enc else np.full(len(F), 10 ** 9)
    # F-44 n-gram EPA (Property 3) with and without prompts
    def epa(with_prompts):
        mk = np.zeros(len(F), bool)
        for i, g in enumerate(FG):
            if g not in tbot: continue
            te = sc['ted'].get(FGs[g]); cand_w = [x for x in (te, tpr.get(g) if with_prompts else None) if x is not None]
            if not cand_w or tbot[g] <= min(cand_w): mk[i:i + N] = True
        return mk
    e1, e2 = epa(True), epa(False)
    # F-44 WikiWho
    x = d[d.user_id == u]
    ev = [(t, 1, 'W', c) for t, c in zip(times, g_.writing_content)] if g_ is not None else []
    ev += [(pd.Timestamp(t), 0, 'Q', s) for t, s in zip(x.user_ask_time, x.user_utterance_text)]
    ev += [(pd.Timestamp(t), 0, 'R', s) for t, s in zip(x.chatbot_utterance_time, x.chatbot_utterance_text)]
    ev.sort(key=lambda e: (e[0], e[1]))
    try: ww, _, _ = wikiwho_chatbot_marks([(k, s) for _, _, k, s in ev], F); ww = np.array(ww, bool); wwfail = False
    except Exception: ww = np.zeros(len(F), bool); wwfail = True
    # F-45 echo that entered after a reply containing it
    after = np.zeros(len(F), bool)
    for i, g in enumerate(FG):
        if cls[g] == 'E':
            te = sc['ted'].get(FGs[g])
            if te is not None and g in tbot and te >= tbot[g]: after[i:i + N] = True
    Eafter = (lab == 'E') & after
    # F-46
    onlyspecial = np.zeros(len(F), bool); coverC = np.zeros(len(F), bool)
    for i, g in enumerate(FG):
        if cls[g] == 'C':
            f46['C_ngrams'] += 1; f46['numeric'] += is_num(g); f46['function'] += is_func(g); coverC[i:i + N] = True
    plain = np.zeros(len(F), bool)
    for i, g in enumerate(FG):
        if cls[g] == 'C' and not (is_num(g) or is_func(g)): plain[i:i + N] = True
    cm = lab == 'C'; f46['C_tokens'] += int(cm.sum()); f46['C_tokens_only_numeric_or_function'] += int((cm & ~plain).sum())
    sh = lambda m: float(m.mean()) if len(F) else 0.0
    rows.append(dict(u=u, C=sh(cm), epa=sh(e1), epa_np=sh(e2), ww=sh(ww), wwfail=wwfail, E=sh(lab == 'E'), Elarge=sh((lab == 'E') & (size > 5)),
                     Eafter=sh(Eafter), Eafter_large=sh(Eafter & (size > 5)), ntok=len(F)))
    agreeC.append((int((e1 == cm).sum()), int((ww == cm).sum()), int((ww & cm).sum()), int(ww.sum()), int(cm.sum()), len(F)))
    if kk % 25 == 0: log(kk, rows[-1])
S = pd.DataFrame(rows).set_index('u'); A_ = np.array(agreeC)
def thr(v, t):
    v = np.asarray(v); b = [np.mean(v[rng.integers(0, len(v), len(v))] >= t) for _ in range(NB)]
    return [float(np.mean(v >= t)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def summ(v):
    return {'mean': float(v.mean()), 'median': float(v.median()), **{f'at_or_above_{int(t * 100)}': thr(v, t) for t in (.25, .5, .75)},
            'spearman_with_U': float(spearmanr(v, S.C).correlation)}
out = {'students': int(len(S)), 'wikiwho_failures': int(S.wwfail.sum())}
out['F44_epa'] = {'U': summ(S.C), 'epa_ngram_two_authors': summ(S.epa), 'epa_ngram_without_prompts': summ(S.epa_np), 'epa_wikiwho': summ(S.ww),
                  'token_agreement_epa_ngram_with_C': float(A_[:, 0].sum() / A_[:, 5].sum()),
                  'token_agreement_wikiwho_with_C': float(A_[:, 1].sum() / A_[:, 5].sum()),
                  'wikiwho_chatbot_tokens_also_C': float(A_[:, 2].sum() / max(A_[:, 3].sum(), 1)), 'C_tokens_also_wikiwho_chatbot': float(A_[:, 2].sum() / A_[:, 4].sum())}
log('F44', {k: round(v['at_or_above_50'][0], 3) for k, v in out['F44_epa'].items() if isinstance(v, dict)})
out['F45_echo_split'] = {'mean_E': float(S.E.mean()), 'mean_E_entered_after_reply': float(S.Eafter.mean()), 'mean_large_E': float(S.Elarge.mean()),
                         'mean_large_E_entered_after_reply': float(S.Eafter_large.mean()),
                         'share_of_large_E_entered_after_reply_tokens': float((S.Eafter_large * S.ntok).sum() / (S.Elarge * S.ntok).sum()),
                         'C_plus_E_after_reply': summ(S.C + S.Eafter), 'C_plus_large_E_after_reply': summ(S.C + S.Eafter_large),
                         'C_plus_large_E_not_after_reply': summ(S.C + S.Elarge - S.Eafter_large)}
log('F45', {k: (round(v['at_or_above_50'][0], 3) if isinstance(v, dict) else round(v, 4)) for k, v in out['F45_echo_split'].items()})
out['F46_numeric_function_ngrams'] = {**f46, 'share_numeric': f46['numeric'] / f46['C_ngrams'], 'share_function': f46['function'] / f46['C_ngrams'],
                                      'share_C_tokens_only_special': f46['C_tokens_only_numeric_or_function'] / f46['C_tokens']}
log('F46', out['F46_numeric_function_ngrams'])

# ---------- F-47 coincidence with replies to similar prompts
pairs = [(u, str(q), str(r)) for u, q, r in zip(d.user_id, d.user_utterance_text, d.chatbot_utterance_text) if u in set(users)]
PU = np.array([p[0] for p in pairs]); vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); M = vec.fit_transform([p[1] for p in pairs])
RG = [set(grams(tok(p[2]), N)) for p in pairs]
byu = {u: np.flatnonzero(PU == u) for u in users}
cov_sim, cov_sim_noH, cov_rand, cov_rand_noH = [], [], [], []
rr = np.random.default_rng(2201)
for u in users:
    idx = byu[u]; F = P[u]; FG = grams(F, N)
    if not len(idx) or not FG: continue
    S_ = (M[idx] @ M.T).toarray(); S_[:, PU == u] = -1; best = S_.argmax(1)
    gs = set().union(*[RG[j] for j in best])
    def cov(gset, noH):
        m = np.zeros(len(F), bool)
        for i, g in enumerate(FG):
            if g in gset and not (noH and (g in H[N] or Cc[N][g] - 1 >= 5)): m[i:i + N] = True
        return m.mean()
    cov_sim.append(cov(gs, False)); cov_sim_noH.append(cov(gs, True))
    cr, crh = [], []
    others = [v for v in users if v != u and len(byu[v])]
    for _ in range(5):
        v = others[rr.integers(len(others))]; gr = set().union(*[RG[j] for j in byu[v]]); cr.append(cov(gr, False)); crh.append(cov(gr, True))
    cov_rand.append(np.mean(cr)); cov_rand_noH.append(np.mean(crh))
out['F47_similar_prompt_coincidence'] = {'students': len(cov_sim), 'mean_coverage_similar_prompt_replies': float(np.mean(cov_sim)),
    'mean_coverage_similar_prompt_replies_without_shared': float(np.mean(cov_sim_noH)), 'mean_coverage_random_other_student': float(np.mean(cov_rand)),
    'mean_coverage_random_other_student_without_shared': float(np.mean(cov_rand_noH)), 'median_coverage_similar_prompt_replies_without_shared': float(np.median(cov_sim_noH)),
    'p95_coverage_similar_prompt_replies_without_shared': float(np.percentile(cov_sim_noH, 95)),
    'matched_prompts_per_student_median': float(np.median([len(byu[u]) for u in users if len(byu[u])]))}
log('F47', out['F47_similar_prompt_coincidence'])
out['seconds'] = round(time.perf_counter() - t00)
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r20_flora.json'), 'w'), indent=1)
log('done')
