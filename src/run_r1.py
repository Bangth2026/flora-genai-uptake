"""Revision R1 analyses (PREREG_R1.md).
Usage: python -I src/run_r1.py DATA_DIR
DATA_DIR: dialogue.xlsx, proposal.xlsx, bio.xlsx, prior.xlsx, post.xlsx, writing.pkl, trace.pkl, annot.pkl"""
import sys, json, re, warnings
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
warnings.filterwarnings('ignore')
D = sys.argv[1]; rng = np.random.default_rng(2027)
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
grams = lambda t, n: [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def mask(P, G, n):
    m = np.zeros(len(P), bool)
    for i, g in enumerate(grams(P, n)):
        if g in G: m[i:i + n] = True
    return m
cov = lambda P, G, n: float(mask(P, G, n).mean()) if len(P) else 0.0
first = lambda df: df.drop_duplicates('user_id', keep='first').set_index('user_id')
lvl = lambda x: float(re.match(r"\s*(\d)", str(x)).group(1)) if re.match(r"\s*(\d)", str(x)) else np.nan
q = lambda v: {k: float(x) for k, x in zip(['p25', 'median', 'p75', 'p90'], np.percentile(v, [25, 50, 75, 90]))}
def bshare(v, thr, B=2000):
    s = [np.mean(rng.choice(v, len(v)) >= thr) for _ in range(B)]
    return {'est': float(np.mean(v >= thr)), 'ci': [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}

d = pd.read_excel(f'{D}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
p = pd.read_excel(f'{D}/proposal.xlsx').set_index('user_id')
bio, pri, post = (first(pd.read_excel(f'{D}/{f}.xlsx')) for f in ['bio', 'prior', 'post'])
w = pd.read_pickle(f'{D}/writing.pkl'); w['t'] = pd.to_datetime(w.writing_log_time, unit='ms'); w = w.sort_values(['user_id', 't'])
tr = pd.read_pickle(f'{D}/trace.pkl'); an = pd.read_pickle(f'{D}/annot.pkl')
users = sorted(set(d.user_id) & set(p.index)); N = len(users)
P = {u: tok(p.loc[u, 'final_proposal']) for u in users}
S = {'n': N}

# third-source sets (M2)
H = {n: set(g for t in an.highlight_text.dropna() for g in grams(tok(t), n)) for n in (5, 8, 12)}
Ccount = {n: Counter(g for u in users for g in set(grams(P[u], n))) for n in (5, 8, 12)}
pastes = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY'].copy()
pastes['txt'] = pastes.event_value.astype(str).str.replace('PASTE_TEXT:::', '', regex=False)
pastes['t'] = pd.to_datetime(pastes.log_trace_time, unit='ms')

def student(u, n, delta_h=0):
    x = d[d.user_id == u]; Pu = P[u]; pg = set(grams(Pu, n))
    tbot, tpr = {}, {}
    for t, txt in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(txt), n): tbot.setdefault(g, t)
    for t, txt in zip(x.user_ask_time, x.user_utterance_text):
        for g in grams(tok(txt), n): tpr.setdefault(g, t)
    B = set(tbot); E_old = set(tpr)
    echo_new = {g for g in B if g in tpr and tpr[g] < tbot[g]}
    third = {g for g in pg if g in H[n] or Ccount[n][g] - 1 >= 5}
    cand = {g for g in pg if g in B}
    ted, jump = {}, {}
    rem = set(cand); prev = 0; strs = {g: ' ' + ' '.join(g) + ' ' for g in cand}
    ws = w[w.user_id == u]; last = ws.t.max() if len(ws) else pd.NaT
    for t, c in zip(ws.t, ws.writing_content):
        if not rem: break
        tt = tok(c); s = ' ' + ' '.join(tt) + ' '
        hit = [g for g in rem if strs[g] in s]
        for g in hit: ted[g] = t; jump[g] = len(tt) - prev
        rem -= set(hit); prev = len(tt)
    off = pd.Timedelta(hours=delta_h)
    def cf(gs, liberal=False):
        return {g for g in gs if (g in ted and tbot[g] <= ted[g] + off) or (liberal and g not in ted)}
    base = cand - echo_new
    out = dict(U=cov(Pu, B, n), Ue_old=cov(Pu, B - E_old, n),
               Uprov_old=cov(Pu, {g for g in cand - E_old if g not in ted or tbot[g] <= ted[g]}, n),
               U_R1=cov(Pu, cf(base - third), n), U_noM2=cov(Pu, cf(base), n), U_M3lib=cov(Pu, cf(base - third, True), n),
               Ue_new=cov(Pu, base, n), Ue_new_M2=cov(Pu, base - third, n))
    out['echo_reply_first'] = sum(1 for g in (B & E_old) if tbot[g] <= tpr[g])
    out['echo_reply_first_in_P'] = sum(1 for g in (cand & E_old) if tbot[g] <= tpr[g])
    out['echo_old_in_P'] = len(cand & E_old)
    G1 = cf(base - third); out['G1'] = G1; out['ted'] = ted; out['jump'] = jump; out['last'] = last
    out['unresolved'] = len((base - third) - set(ted)); out['n_base_third'] = len(base - third)
    out['third_removed_tokens'] = cov(Pu, cf(base), n) - out['U_R1']
    return out

R = {u: student(u, 8) for u in users}
R1h = {u: student(u, 8, 1)['U_R1'] for u in users}
R5 = {u: student(u, 5)['U_R1'] for u in users}; R12 = {u: student(u, 12)['U_R1'] for u in users}
A = {k: np.array([R[u][k] for u in users]) for k in ['U', 'Ue_old', 'Uprov_old', 'U_R1', 'U_noM2', 'U_M3lib', 'Ue_new', 'Ue_new_M2']}
A['U_R1_plus1h'] = np.array([R1h[u] for u in users]); A['U_R1_n5'] = np.array([R5[u] for u in users]); A['U_R1_n12'] = np.array([R12[u] for u in users])
U1 = A['U_R1']; np.save('results/U_R1.npy', U1); np.save('results/U_naive.npy', A['U'])

# ---------- measure changes ----------
S['M1'] = {'reply_first_then_prompt_grams_total': int(sum(R[u]['echo_reply_first'] for u in users)),
           'reply_first_then_prompt_grams_in_proposals': int(sum(R[u]['echo_reply_first_in_P'] for u in users)),
           'old_echo_grams_in_proposals': int(sum(R[u]['echo_old_in_P'] for u in users))}
S['M3'] = {'unresolved_share_of_candidates': float(sum(R[u]['unresolved'] for u in users) / max(1, sum(R[u]['n_base_third'] for u in users)))}
S['versions'] = {k: {**q(v), 'mean': float(v.mean()), 'ge25': bshare(v, .25), 'ge50': bshare(v, .5)} for k, v in A.items()}
S['spearman'] = {k: float(spearmanr(U1, v)[0]) for k, v in A.items() if k != 'U_R1'}

# ---------- V1/V2 paste validation ----------
v_tok, v_cov, logged_cf, logged_all, agree = 0, 0, 0, 0, Counter()
pst_users = sorted(set(pastes.user_id) & set(users))
for u in pst_users:
    pg = set(g for t in pastes[pastes.user_id == u].txt for g in grams(tok(t), 8))
    m1 = mask(P[u], R[u]['G1'], 8); m2 = mask(P[u], R[u]['G1'] & pg, 8)
    v_tok += m1.sum(); v_cov += m2.sum()
    x = d[d.user_id == u]
    tb = {}
    for t, txt in zip(x.chatbot_utterance_time, x.chatbot_utterance_text):
        for g in grams(tok(txt), 8): tb.setdefault(g, t)
    for t, txt in zip(pastes[pastes.user_id == u].t, pastes[pastes.user_id == u].txt):
        for g in set(grams(tok(txt), 8)):
            logged_all += 1; logged_cf += int(g in tb and tb[g] <= t)
    for g in R[u]['G1']:
        if g in R[u]['jump']: agree[(R[u]['jump'][g] >= 20, g in pg)] += 1
S['V1'] = {'n_students_with_logged_pastes': len(pst_users), 'share_UR1_tokens_in_logged_pastes': float(v_cov / max(v_tok, 1)),
           'share_logged_paste_grams_chatbot_first': float(logged_cf / max(logged_all, 1))}
S['V2'] = {f'jump{int(a)}_logged{int(b)}': int(c) for (a, b), c in agree.items()}
S['paste_rule_share_primary'] = float(sum(1 for u in users for g in R[u]['G1'] if R[u]['jump'].get(g, 0) >= 20) / max(1, sum(len(R[u]['G1']) for u in users)))

# ---------- V3 chance ----------
botf = {}
for u in users:
    x = d[d.user_id == u]; B = set(g for t in x.chatbot_utterance_text for g in grams(tok(t), 8))
    botf[u] = {g for g in B if g not in H[8]}
obs_minus, above = [], []
for i, u in enumerate(users):
    others = [v for v in users if v != u]; pgs = set(grams(P[u], 8))
    nul = np.array([cov(P[u], {g for g in botf[v] & pgs if Ccount[8][g] - 1 < 5}, 8) for v in rng.choice(others, 200)])
    obs_minus.append(U1[i] - np.median(nul)); above.append(U1[i] > np.percentile(nul, 99))
obs_minus, above = np.array(obs_minus), np.array(above)
S['V3'] = {'chance_corrected': q(obs_minus), 'share_above_own_null99': bshare(above.astype(float), 1)}

# ---------- RQ1 timing and position ----------
late24 = late2 = tot = 0; thirds = np.zeros(3)
for u in users:
    Pu = P[u]; G1 = R[u]['G1']; ted = R[u]['ted']; last = R[u]['last']
    tmin = [None] * len(Pu)
    for i, g in enumerate(grams(Pu, 8)):
        if g in G1:
            for j in range(i, i + 8):
                tmin[j] = ted[g] if tmin[j] is None or ted[g] < tmin[j] else tmin[j]
    for j, t in enumerate(tmin):
        if t is None: continue
        tot += 1; h = (last - t).total_seconds() / 3600
        late24 += h <= 24; late2 += h <= 2; thirds[min(2, int(3 * j / len(Pu)))] += 1
S['timing'] = {'share_tokens_last24h': late24 / max(tot, 1), 'share_tokens_last2h': late2 / max(tot, 1),
               'share_by_third': (thirds / thirds.sum()).tolist()}

# ---------- RQ2 ----------
thr = float(np.percentile(U1, 75)); y = (U1 >= thr).astype(int); S['rq2_label'] = {'threshold': thr, 'n_pos': int(y.sum())}
codes = d.user_utterance_code.value_counts(); keep = [c for c in codes.index if codes[c] >= 20 and c != 'Others']
KW = re.compile(r"\b(write|draft|rewrite|paragraph|section|introduction|conclusion|proposal|improve|polish|paraphrase)\b", re.I)
HEAD = re.compile(r"(^|\n)\s*(#{1,6}\s|\*\*[^*\n]+\*\*)")
F = []
for u in users:
    x = d[d.user_id == u]; n = len(x); last = R[u]['last']
    r = {f'sh_{c}': (x.user_utterance_code == c).sum() / n for c in keep}
    r['log_turns'] = np.log(n + 1)
    r['log_bot_words'] = np.log(sum(len(tok(t)) for t in x.chatbot_utterance_text) + 1)
    r['kw_share'] = x.user_utterance_text.astype(str).str.contains(KW).mean()
    r['reply_len'] = np.mean([np.log(len(tok(t)) + 1) for t in x.chatbot_utterance_text])
    r['head_share'] = x.chatbot_utterance_text.astype(str).str.contains(HEAD).mean()
    r['late_share'] = float(((last - x.user_ask_time).dt.total_seconds() <= 86400).mean()) if pd.notna(last) else 0.0
    F.append(r)
F = pd.DataFrame(F, index=users)
intent = [c for c in F.columns if c.startswith('sh_')] + ['log_turns']
reqf = ['kw_share', 'reply_len', 'head_share', 'late_share', 'log_turns']
sets = {'exposure': ['log_bot_words', 'log_turns'], 'intent': intent, 'request': reqf,
        'intent+exposure': intent + ['log_bot_words'], 'request+exposure': reqf + ['log_bot_words'],
        'all': list(dict.fromkeys(intent + reqf + ['log_bot_words']))}
def oof(cols, seeds=20):
    X = F[cols].values; o = np.zeros((seeds, N))
    for s in range(seeds):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            sc = StandardScaler().fit(X[a]); o[s, b] = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[a]), y[a]).predict_proba(sc.transform(X[b]))[:, 1]
    return o.mean(0)
O = {k: oof(v) for k, v in sets.items()}
def boot(fn, B=2000):
    vals = []
    for _ in range(B):
        b = rng.choice(N, N)
        if y[b].min() != y[b].max(): vals.append(fn(b))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
S['rq2'] = {k: {'auc': float(roc_auc_score(y, o)), 'ci': boot(lambda b, o=o: roc_auc_score(y[b], o[b]))} for k, o in O.items()}
for k in ['intent+exposure', 'request+exposure', 'all']:
    g = roc_auc_score(y, O[k]) - roc_auc_score(y, O['exposure'])
    ci = boot(lambda b, k=k: roc_auc_score(y[b], O[k][b]) - roc_auc_score(y[b], O['exposure'][b]))
    S['rq2'][k]['gain'] = {'est': float(g), 'ci': ci, 'verdict': 'adds' if ci[0] > 0 else ('negligible' if ci[1] < .02 else 'inconclusive')}

# ---------- RQ3 ----------
score = p.reindex(users).proposal_score.values.astype(float)
C = pd.DataFrame(index=users)
C['prior_ds'] = pri.reindex(users).has_prior_knowledge_in_DS
C['genai_fam'] = pri.reindex(users).extent_of_familiar_genai.map(lvl)
mt = bio.reindex(users).mother_tongue
C['english_l1'] = mt.astype(str).str.strip().str.lower().eq('english').astype(float).where(mt.notna())
C['master'] = bio.reindex(users).degree.astype(str).str.contains('Master').astype(float).where(bio.reindex(users).degree.notna())
S['imputed'] = {c: int(C[c].isna().sum()) for c in C.columns}; C = C.fillna(C.median())
Z = pd.DataFrame({'U_z': (U1 - U1.mean()) / U1.std()}, index=users)
def ols(X):
    f = sm.OLS(score, sm.add_constant(X)).fit(cov_type='HC3')
    return {'b': float(f.params['U_z']), 'ci': [float(v) for v in f.conf_int().loc['U_z']], 'p': float(f.pvalues['U_z'])}
S['rq3'] = {'a_unadjusted': ols(Z), 'b_pretask': ols(Z.join(C)),
            'c_full': ols(Z.join(C).assign(log_len=np.log([len(P[u]) for u in users]), log_bot=F.log_bot_words.values))}
hi = U1 >= .5
diffs = []
for _ in range(2000):
    b = rng.choice(N, N)
    if hi[b].any() and (~hi[b]).any(): diffs.append(score[b][hi[b]].mean() - score[b][~hi[b]].mean())
S['rq3']['ge50_minus_rest'] = {'n_ge50': int(hi.sum()), 'est': float(score[hi].mean() - score[~hi].mean()), 'ci': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
qt = pd.qcut(pd.Series(U1).rank(method='first'), 4, labels=False)
S['rq3']['score_by_quartile'] = [float(score[qt.values == k].mean()) for k in range(4)]
und = post.reindex(users).extent_understanding_of_subject_improved.map(lvl); ok = und.notna().values
r = spearmanr(U1[ok], und.values[ok]); S['rq3']['understanding'] = {'n': int(ok.sum()), 'rho': float(r[0]), 'p': float(r[1])}
S['rq3']['responders_vs_non'] = {'mean_resp': float(U1[ok].mean()), 'mean_non': float(U1[~ok].mean()), 'ci_diff': boot(lambda b: U1[b][ok[b]].mean() - U1[b][~ok[b]].mean())}

# ---------- RQ4 ----------
eng = C.english_l1.values == 1
S['rq4'] = {'n_english': int(eng.sum()), 'mean_eng': float(U1[eng].mean()), 'mean_other': float(U1[~eng].mean()),
            'median_eng': float(np.median(U1[eng])), 'median_other': float(np.median(U1[~eng])),
            'ge25_eng': float((U1[eng] >= .25).mean()), 'ge25_other': float((U1[~eng] >= .25).mean()),
            'diff_mean_other_minus_eng_ci': boot(lambda b: U1[b][~eng[b]].mean() - U1[b][eng[b]].mean())}
json.dump(S, open('results/r1_stats.json', 'w'), indent=1); print(json.dumps(S, indent=1))
