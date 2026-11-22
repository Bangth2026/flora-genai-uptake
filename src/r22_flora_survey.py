"""Analysis log R22, F-50: FLoRA post-task survey (file 5.7_post-experience_using_platform_survey.xlsx), free-text answers
to the question on other tools ('use_other_tools'). An answer counts as naming an outside AI chatbot if it names
ChatGPT, GPT-3.5, Claude, Copilot, Gemini, Bard, Perplexity or another chatbot by name; answers that refer only to the
chatbot inside FLoRA ('GPT-powered chatbot in FLoRA', 'GPT-powered mentions') or to a search engine (Google, Bing) do
not count. The answers were unprompted free text, so the count is a lower bound on outside chatbot use and says nothing
about copying the embedded model's wording. Mean class shares of students who named one and of the other
respondents are given for description only. Usage: python -I src/r22_flora_survey.py POST_SURVEY_XLSX IDS_CSV
-> results/r22_flora_survey.json"""
import sys, re, json, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
x = pd.read_excel(sys.argv[1]); ids = set(pd.read_csv(sys.argv[2]).iloc[:, 0])
s = x.use_other_tools.astype(str)
NAMED = re.compile(r'chat ?-?gpt|gpt[- ]?3|gpt[- ]?4|claude|copilot|gemini|bard|perplexity', re.I)
INSIDE = re.compile(r'gpt-powered', re.I)
hit = s.str.contains(NAMED) & ~s.str.contains(INSIDE)
out = {'respondents': int(len(x)), 'respondents_in_275': int(x.user_id.isin(ids).sum()),
       'name_outside_chatbot': int(hit.sum()), 'share': float(hit.mean()),
       'answers_counted': [v for v in s[hit]]}
sc = pd.read_csv(os.path.join(HERE, '..', 'results', 'flora_student_classes.csv')).set_index('user_id')
named = set(x.user_id[hit]); other = set(x.user_id[~hit])
for k, grp in (('named', named), ('not_named', other)):
    g = sc.loc[sorted(grp & set(sc.index))]
    out[f'classes_{k}'] = {'n': int(len(g)), **{c: float(g[c].mean()) for c in ('C', 'E', 'S', 'T', 'X')}, 'X_with_C': float(g.X_with_C.mean())}
json.dump(out, open(os.path.join(HERE, '..', 'results', 'r22_flora_survey.json'), 'w'), indent=1)
print(out['classes_named'], out['classes_not_named']); print(out['respondents'], out['respondents_in_275'], out['name_outside_chatbot'], round(out['share'], 3))
for v in out['answers_counted']: print('-', v[:150])
