"""Analysis log R28, F-62 to F-65 (FLoRA): the process-evidence analyses that read the system-log trace, rerun on the
whole trace (DATA_FULL/trace.pkl = all eight sheets, written by src/r28_read_trace.py) instead of the first sheet of each
workbook. The scripts are executed unchanged except for their output paths; nothing in the attribution changes.
F-62 src/r10_flora_checks.py (P-1 share of each class in logged pasted text, P-2 editor-log completeness)
F-63 src/r12_flora.py (F-10 logged pastes dominated by C against those dominated by X)
F-64 src/r23_flora.py, sections F-52 (data-driven tau), F-53 (census of chatbot-first runs) and F-54 (the seven passages)
F-65 trace coverage: students with editor keystrokes, logged pastes and chatbot-panel selections
Usage: python -I src/r28_process.py DATA_FULL_DIR STEP   (STEP: f62, f63, f64, f65) -> results/r28_process_STEP.json"""
import sys, os, json, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
D, STEP = sys.argv[1], sys.argv[2]
os.chdir(ROOT)
outp = f'results/r28_process_{STEP}.json'

def run_patched(script, replacements, argv):
    src = open(os.path.join(HERE, script)).read()
    for a, b in replacements:
        assert src.count(a) >= 1, (script, a); src = src.replace(a, b)
    sys.argv = [os.path.join(HERE, script)] + argv
    G = {'__name__': '__main__', '__file__': os.path.join(HERE, script)}
    exec(compile(src, script, 'exec'), G)
    return G

if STEP == 'f62':
    run_patched('r10_flora_checks.py', [("'results/r10_flora_checks.json'", repr(outp))], [D])
elif STEP == 'f63':
    run_patched('r12_flora.py', [("'results/r12_flora.json'", repr(outp))], [D])
elif STEP == 'f64':
    src = open(os.path.join(HERE, 'r23_flora.py')).read()
    a = src.index('# ---------- F-51 partial identification'); b = src.index('# ---------- F-52 data-driven tau')
    src = src[:a] + src[b:]                       # F-51 (labels only, no trace) is not repeated
    src = src.replace("os.path.join(HERE, '..', 'results', 'r23_flora.json')", repr(os.path.join(ROOT, outp)))
    sys.argv = [os.path.join(HERE, 'r23_flora.py'), D]
    G = {'__name__': '__main__', '__file__': os.path.join(HERE, 'r23_flora.py')}
    exec(compile(src, 'r23_flora.py', 'exec'), G)
elif STEP == 'f65':
    import pandas as pd
    tr = pd.read_pickle(f'{D}/trace.pkl')
    key = tr[(tr.learning_action_label == 'WRITE_ESSAY') & tr.event_value.astype(str).str.startswith('KEY:::')]
    pas = tr[tr.learning_action_label == 'PASTE_TEXT_ESSAY']
    sel = tr[(tr.learning_action_label == 'READ_FEEDBACK_MULTI_AGENTS') & tr.event_value.astype(str).str.startswith('MOUSE_SELECT:::')]
    o = {'students_in_trace': int(tr.user_id.nunique()), 'with_editor_keystrokes': int(key.user_id.nunique()),
         'with_logged_pastes': int(pas.user_id.nunique()), 'with_panel_selections': int(sel.user_id.nunique()),
         'paste_events': int(len(pas)), 'keystroke_events': int(len(key)), 'selection_events': int(len(sel))}
    json.dump(o, open(outp, 'w'), indent=1); print(o)
