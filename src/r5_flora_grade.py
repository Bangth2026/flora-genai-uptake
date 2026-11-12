"""Plan R5, F7: robustness of the FLoRA grade analysis (missing covariates, range, transformations of U*, class shares).
Usage: python -I src/r5_flora_grade.py DATA_DIR -> results/r5_flora_grade.json (needs results/U_R1.npy, results/r5_flora_classes.csv)"""
import sys, re, json, warnings
import numpy as np, pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.imputation import mice
warnings.filterwarnings('ignore')
D = sys.argv[1]
TOK = re.compile(r"[a-z0-9]+"); tok = lambda s: TOK.findall(str(s).lower())
first = lambda df: df.drop_duplicates('user_id', keep='first').set_index('user_id')
lvl = lambda x: float(re.match(r"\s*(\d)", str(x)).group(1)) if re.match(r"\s*(\d)", str(x)) else np.nan
d = pd.read_excel(f'{D}/dialogue.xlsx'); p = pd.read_excel(f'{D}/proposal.xlsx').set_index('user_id')
bio, pri = (first(pd.read_excel(f'{D}/{f}.xlsx')) for f in ['bio', 'prior'])
users = sorted(set(d.user_id) & set(p.index)); U = np.load('results/U_R1.npy')
cls = pd.read_csv('results/r5_flora_classes.csv', index_col=0).loc[users]
df = pd.DataFrame(index=users)
df['score'] = p.reindex(users).proposal_score.values.astype(float)
df['U'] = U; df['Uz'] = (U - U.mean()) / U.std()
df['prior_ds'] = pri.reindex(users).has_prior_knowledge_in_DS
df['genai_fam'] = pri.reindex(users).extent_of_familiar_genai.map(lvl)
mt = bio.reindex(users).mother_tongue
df['english_l1'] = mt.astype(str).str.strip().str.lower().eq('english').astype(float).where(mt.notna())
df['master'] = bio.reindex(users).degree.astype(str).str.contains('Master').astype(float).where(bio.reindex(users).degree.notna())
df['prior_ds'] = pd.to_numeric(df.prior_ds, errors='coerce')
df['log_len'] = np.log([len(tok(p.loc[u, 'final_proposal'])) for u in users])
df['log_bot'] = [np.log(sum(len(tok(t)) for t in d[d.user_id == u].chatbot_utterance_text) + 1) for u in users]
for c in 'CETX': df[f'{c}z'] = (cls[c] - cls[c].mean()) / cls[c].std()
COV = ['prior_ds', 'genai_fam', 'english_l1', 'master']
out = {'n': len(df), 'missing': {c: int(df[c].isna().sum()) for c in COV}, 'missing_any': int(df[COV].isna().any(axis=1).sum())}
def ols(formula, data):
    f = smf.ols(formula, data).fit(cov_type='HC3'); return f
def coef(f, v): return [float(f.params[v]), [float(c) for c in f.conf_int().loc[v]], float(f.pvalues[v])]
med = df.copy(); med[COV] = med[COV].fillna(med[COV].median())
fb = 'score ~ Uz + ' + ' + '.join(COV); fc = fb + ' + log_len + log_bot'
out['median_imputed'] = {'b': coef(ols(fb, med), 'Uz'), 'c': coef(ols(fc, med), 'Uz')}
mi = df.copy()
for c in COV: mi[f'{c}_miss'] = mi[c].isna().astype(float); mi[c] = mi[c].fillna(0)
miss_terms = [f'{c}_miss' for c in COV if mi[f'{c}_miss'].sum() > 0]
out['missing_indicator'] = {'b': coef(ols(fb + ''.join(' + ' + t for t in miss_terms), mi), 'Uz'), 'c': coef(ols(fc + ''.join(' + ' + t for t in miss_terms), mi), 'Uz')}
for nm, fml in [('b', fb), ('c', fc)]:
    cols = ['score', 'Uz', 'log_len', 'log_bot'] + COV
    imp = mice.MICEData(df[cols].reset_index(drop=True).copy())
    np.random.seed(2052)
    r = mice.MICE(fml, sm.OLS, imp).fit(n_burnin=10, n_imputations=20)
    i = list(r.model.exog_names).index('Uz') if hasattr(r, 'model') else 1
    names = r.exog_names if hasattr(r, 'exog_names') else None
    b = float(r.params[1]); se = float(r.bse[1])
    out.setdefault('mice', {})[nm] = [b, [b - 1.96 * se, b + 1.96 * se], float(2 * stats.norm.sf(abs(b / se)))]
# grade distribution and censored model
vc = df.score.value_counts().sort_index()
out['grade_distribution'] = {str(k): int(v) for k, v in vc.items()}
out['grade_share_15'] = float((df.score >= 15).mean()); out['grade_share_ge14'] = float((df.score >= 14).mean())
out['grade_min'] = float(df.score.min()); out['grade_skew'] = float(stats.skew(df.score))
y = df.score.values; cen = y >= 15; X = np.c_[np.ones(len(df)), df.Uz.values]
def nll(th):
    b, s = th[:2], np.exp(th[2]); mu = X @ b
    return -(stats.norm.logpdf(y[~cen], mu[~cen], s).sum() + stats.norm.logsf(15, mu[cen], s).sum())
f = optimize.minimize(nll, np.r_[np.linalg.lstsq(X, y, rcond=None)[0], 0.0], method='BFGS'); se = np.sqrt(np.diag(f.hess_inv))
out['tobit15'] = [float(f.x[1]), [float(f.x[1] - 1.96 * se[1]), float(f.x[1] + 1.96 * se[1])], float(2 * stats.norm.sf(abs(f.x[1] / se[1])))]
# transformations of U*
df['Urank'] = (df.U.rank() / len(df) - .5) / (df.U.rank() / len(df)).std()
df['Ulog'] = np.log(df.U + .01); df['Ulog'] = (df.Ulog - df.Ulog.mean()) / df.Ulog.std()
out['transform'] = {'rank_per_sd': coef(ols('score ~ Urank', df), 'Urank'), 'log_per_sd': coef(ols('score ~ Ulog', df), 'Ulog')}
fs = smf.ols('score ~ bs(U, df=3)', df).fit(cov_type='HC3'); names = [n for n in fs.params.index if n.startswith('bs(')]
wt = fs.wald_test(' , '.join(f'{n} = 0' for n in names), scalar=True)
out['transform']['spline_wald'] = {'stat': float(wt.statistic), 'p': float(wt.pvalue), 'df': len(names)}
# class shares
med2 = med.copy()
for c in 'XTE':
    out.setdefault('class_shares', {})[c] = {'unadjusted': coef(ols(f'score ~ {c}z', med2), f'{c}z'), 'adjusted': coef(ols(f'score ~ {c}z + ' + ' + '.join(COV), med2), f'{c}z')}
out['class_shares']['joint_CTX'] = {k: coef(ols('score ~ Cz + Tz + Xz + ' + ' + '.join(COV), med2), k) for k in ['Cz', 'Tz', 'Xz']}
json.dump(out, open('results/r5_flora_grade.json', 'w'), indent=1); print(json.dumps(out, indent=1))
