"""Analysis log R28 (data reading). The FLoRA system-log trace is released as two workbooks (trace1.xlsx, trace2.xlsx), each
with four sheets ('Result 1' to 'Result 4'; 7,542,124 rows in total). Every earlier entry read the trace as DATA/trace.pkl,
which held only the first sheet of each workbook (2,071,788 rows, 71 students); the editor keystrokes, logged pastes and
chatbot-panel selections of the other students were therefore never read. This script reads all eight sheets and writes
DATA/trace_full.pkl with the same columns; DATA/trace.pkl is left unchanged so that earlier entries can be rerun.
Usage: python -I src/r28_read_trace.py DATA_DIR"""
import sys, json
import pandas as pd

D = sys.argv[1]
parts, rows = [], {}
for f in ('trace1.xlsx', 'trace2.xlsx'):
    for sh in ('Result 1', 'Result 2', 'Result 3', 'Result 4'):
        x = pd.read_excel(f'{D}/{f}', sheet_name=sh, engine='calamine')
        rows[f'{f}:{sh}'] = len(x); parts.append(x); print(f, sh, len(x), x.user_id.nunique(), flush=True)
t = pd.concat(parts, ignore_index=True); del parts
for c in ('url', 'source', 'learning_action_label', 'action_category_label'):
    if c in t: t[c] = t[c].astype('category')
t.to_pickle(f'{D}/trace_full.pkl')
first = pd.read_pickle(f'{D}/trace.pkl')
info = {'rows_by_sheet': rows, 'rows_total': int(len(t)), 'students_total': int(t.user_id.nunique()),
        'rows_in_trace_pkl': int(len(first)), 'students_in_trace_pkl': int(first.user_id.nunique())}
print(json.dumps(info, indent=1))
json.dump(info, open('results/r28_trace_reading.json', 'w'), indent=1)
