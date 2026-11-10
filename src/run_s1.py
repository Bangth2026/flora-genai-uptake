"""Study 1 (FLoRA): measured uptake of chatbot text. Implements PREREG_S1_flora.md.

Usage: python -I src/run_s1.py --data DATA_DIR --out results/s1_stats.json
DATA_DIR holds the Figshare files renamed: dialogue.xlsx, proposal.xlsx, bio.xlsx, prior.xlsx, post.xlsx
"""
import argparse, json, re, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import statsmodels.api as sm

warnings.filterwarnings('ignore')
TOK = re.compile(r"[a-z0-9]+")
tok = lambda s: TOK.findall(str(s).lower())


def ngrams(t, n):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def coverage(prop, grams, n):
    if len(prop) < n:
        return 0.0
    cov = np.zeros(len(prop), bool)
    for i in range(len(prop) - n + 1):
        if tuple(prop[i:i + n]) in grams:
            cov[i:i + n] = True
    return float(cov.mean())


def first(df):
    return df.drop_duplicates('user_id', keep='first').set_index('user_id')


def lvl(x):
    m = re.match(r"\s*(\d)", str(x))
    return float(m.group(1)) if m else np.nan


def main(a):
    rng = np.random.default_rng(2026)
    d = pd.read_excel(f'{a.data}/dialogue.xlsx').sort_values(['user_id', 'user_ask_time'])
    p = pd.read_excel(f'{a.data}/proposal.xlsx').set_index('user_id')
    bio, pri, post = (first(pd.read_excel(f'{a.data}/{f}.xlsx')) for f in ['bio', 'prior', 'post'])
    users = sorted(set(d.user_id) & set(p.index))
    S = {'n_students': len(users), 'n_turns': int(d[d.user_id.isin(users)].shape[0])}

    # ---------------- uptake ----------------
    prop_tok = {u: tok(p.loc[u, 'final_proposal']) for u in users}
    bot_tok = {u: tok(' '.join(d[d.user_id == u].chatbot_utterance_text.astype(str))) for u in users}
    U = {}
    for n in (5, 8, 12):
        U[n] = np.array([coverage(prop_tok[u], ngrams(bot_tok[u], n), n) for u in users])
    g8 = {u: ngrams(bot_tok[u], 8) for u in users}
    null = np.zeros((len(users), 200))
    for i, u in enumerate(users):
        others = [v for v in users if v != u]
        for k, v in enumerate(rng.choice(others, 200)):
            null[i, k] = coverage(prop_tok[u], g8[v], 8)
    null99 = float(np.percentile(null, 99))
    u8 = U[8] if not a.label else np.load(a.label)
    excess = u8 - np.median(null, axis=1)
    q = lambda x: {k: float(v) for k, v in zip(['p10', 'p25', 'median', 'p75', 'p90', 'max'], np.percentile(x, [10, 25, 50, 75, 90, 100]))}
    S['rq1'] = {'U8': {**q(u8), 'mean': float(u8.mean())}, 'U5_median': float(np.median(U[5])), 'U12_median': float(np.median(U[12])),
                'null99_pooled': null99, 'null_median_of_medians': float(np.median(np.median(null, axis=1))),
                'share_above_null99': float((u8 > null99).mean()), 'n_above_null99': int((u8 > null99).sum()),
                'excess': q(excess), 'share_U_ge_0.10': float((u8 >= .10).mean()), 'share_U_ge_0.25': float((u8 >= .25).mean()),
                'spearman_U5_U12': float(spearmanr(U[5], U[12])[0])}
    thr = float(np.percentile(u8, 75))
    y = (u8 >= thr).astype(int)
    S['label'] = {'top_quartile_threshold': thr, 'n_high': int(y.sum())}

    # ---------------- features ----------------
    counts = d[d.user_id.isin(users)].user_utterance_code.value_counts()
    keep = [c for c in counts.index if counts[c] >= 20 and c != 'Others']

    def feats(dd):
        rows = []
        for u in users:
            x = dd[dd.user_id == u]
            n = max(len(x), 1)
            r = {f'sh_{c}': (x.user_utterance_code == c).sum() / n for c in keep}
            r['sh_Other'] = 1 - sum(r.values())
            r['log_turns'] = np.log(len(x) + 1)
            r['log_bot_words'] = np.log(sum(len(tok(t)) for t in x.chatbot_utterance_text) + 1)
            rows.append(r)
        return pd.DataFrame(rows, index=users)

    F = feats(d)
    rank = d.groupby('user_id').cumcount(); size = d.groupby('user_id').user_id.transform('size')
    half = d[rank < np.ceil(size / 2)]
    Fh = feats(half)
    beh = [c for c in F.columns if c.startswith('sh_') and c != 'sh_Other'] + ['log_turns']
    exp_ = ['log_bot_words', 'log_turns']
    S['codes_used'] = keep

    def cv_oof(X, seeds=20):
        oof = np.zeros((seeds, len(y)))
        for s in range(seeds):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                sc = StandardScaler().fit(X[tr])
                m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                oof[s, te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return oof

    def boot_diff(pa, pb, B=2000):
        idx = np.arange(len(y)); out = []
        for _ in range(B):
            b = rng.choice(idx, len(idx))
            if y[b].min() == y[b].max():
                continue
            out.append(roc_auc_score(y[b], pb[b]) - roc_auc_score(y[b], pa[b]))
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

    def block(FF):
        res = {}
        o = {k: cv_oof(FF[cols].values) for k, cols in [('exposure', exp_), ('behaviour', beh), ('combined', beh + ['log_bot_words'])]}
        for k, v in o.items():
            aucs = [roc_auc_score(y, r) for r in v]
            res[k] = {'auc_mean': float(np.mean(aucs)), 'auc_sd': float(np.std(aucs))}
        m = {k: v.mean(0) for k, v in o.items()}
        dlt = roc_auc_score(y, m['combined']) - roc_auc_score(y, m['exposure'])
        ci = boot_diff(m['exposure'], m['combined'])
        verdict = 'adds' if ci[0] > 0 else ('negligible' if ci[1] < 0.02 else 'inconclusive')
        res['delta_combined_minus_exposure'] = {'est': float(dlt), 'ci': ci, 'verdict': verdict}
        return res, m

    S['rq2'], oofm = block(F)
    S['rq2_first_half'], _ = block(Fh)
    # coefficients of combined model, bootstrap CI
    X = StandardScaler().fit_transform(F[beh + ['log_bot_words']].values)
    fit = lambda Xb, yb: LogisticRegression(C=1.0, max_iter=2000).fit(Xb, yb).coef_[0]
    coef = fit(X, y); bs = []
    for _ in range(2000):
        b = rng.choice(len(y), len(y))
        if y[b].min() != y[b].max():
            bs.append(fit(X[b], y[b]))
    bs = np.array(bs)
    S['rq2_coef'] = {c: {'b': float(coef[i]), 'ci': [float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))]}
                     for i, c in enumerate(beh + ['log_bot_words'])}
    # exposure-uptake correlation
    S['rq2']['spearman_U_botwords'] = float(spearmanr(u8, F.log_bot_words)[0])

    # ---------------- covariates ----------------
    C = pd.DataFrame(index=users)
    C['prior_ds'] = pri.reindex(users).has_prior_knowledge_in_DS
    C['years_exp'] = pri.reindex(users).years_of_experience
    C['genai_fam'] = pri.reindex(users).extent_of_familiar_genai.map(lvl)
    mt = bio.reindex(users).mother_tongue
    C['english_l1'] = mt.astype(str).str.strip().str.lower().eq('english').astype(float).where(mt.notna())
    C['master'] = bio.reindex(users).degree.astype(str).str.contains('Master').astype(float).where(bio.reindex(users).degree.notna())
    C['prop_words'] = np.log([len(prop_tok[u]) for u in users])
    S['imputed'] = {c: int(C[c].isna().sum()) for c in C.columns}
    C = C.fillna(C.median())

    # ---------------- RQ3 ----------------
    score = p.reindex(users).proposal_score.values.astype(float)
    Z = pd.DataFrame({'U_z': (u8 - u8.mean()) / u8.std()}, index=users).join(C).join(F[['log_bot_words']])
    ols = sm.OLS(score, sm.add_constant(Z)).fit(cov_type='HC3')
    S['rq3'] = {'b_U_z': float(ols.params['U_z']), 'ci': [float(x) for x in ols.conf_int().loc['U_z']], 'p': float(ols.pvalues['U_z']),
                'n': int(len(score)), 'r2': float(ols.rsquared), 'b_unadjusted_spearman': float(spearmanr(u8, score)[0])}
    und = post.reindex(users).extent_understanding_of_subject_improved.map(lvl)
    ok = und.notna().values
    r = spearmanr(u8[ok], und.values[ok])
    S['rq3']['understanding'] = {'n': int(ok.sum()), 'rho': float(r[0]), 'p': float(r[1])}

    # ---------------- RQ4 ----------------
    eng = C.english_l1.values == 1
    flag = (oofm['combined'] >= np.percentile(oofm['combined'], 75)).astype(int)
    def bdiff(v):
        out = []
        for _ in range(2000):
            b = rng.choice(len(v), len(v)); e = eng[b]
            if e.sum() and (~e).sum():
                out.append(v[b][~e].mean() - v[b][e].mean())
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
    S['rq4'] = {'n_english_l1': int(eng.sum()), 'U_mean_english': float(u8[eng].mean()), 'U_mean_other': float(u8[~eng].mean()),
                'U_diff_other_minus_english_ci': bdiff(u8), 'flag_rate_english': float(flag[eng].mean()),
                'flag_rate_other': float(flag[~eng].mean()), 'flag_diff_ci': bdiff(flag.astype(float))}
    json.dump(S, open(a.out, 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--out'); ap.add_argument('--label', default=None)
    main(ap.parse_args())
