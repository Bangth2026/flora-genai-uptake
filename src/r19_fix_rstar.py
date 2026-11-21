"""Analysis log R19: the first full run of src/r19_benchmark.py returned r* = 0 for methods that make fewer false
chatbot labels than the attribution (the audit operating point; earliest plausible attribution on some sets), where r* is
not defined. This sets those entries to null in results/r19_benchmark.json; every other value is unchanged. The definition
is corrected in src/r19_benchmark.py. Usage: python -I src/r19_fix_rstar.py"""
import json
p = 'results/r19_benchmark.json'; d = json.load(open(p)); n = 0
for s, o in d['B40_decision_cost'].items():
    cA = o['attribution']['cost_per_1000_tokens']; fpA = (cA['5'] - cA['1']) / 4
    for m, v in o['ratio_above_which_attribution_costs_less'].items():
        c = o[m]['cost_per_1000_tokens']; fp = (c['5'] - c['1']) / 4
        if fp - fpA <= 1e-12: o['ratio_above_which_attribution_costs_less'][m] = None; n += 1
d['B40_note'] = 'r* is null where the method makes no more false chatbot labels than the attribution (src/r19_fix_rstar.py)'
json.dump(d, open(p, 'w'), indent=1); print('set to null:', n)
