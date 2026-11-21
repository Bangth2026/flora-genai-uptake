"""Earliest plausible attribution as WikiWho implements it (Floeck & Acosta, 2014; PyPI package WikiWho 1.0.3):
hierarchical matching of paragraphs and sentences against every earlier revision, and word-level diff against the
previous revision. The two-author revision sequence of Property 3 is fed to it in time order: every editor snapshot is
a revision by the writer; every prompt and every reply is a revision whose text is the current editor text followed by
the prompt (writer) or the reply (chatbot) as a separate paragraph, so that WikiWho's hash tables hold the dialogue text
when it later appears in the editor. WikiWho's spam and vandalism heuristics, written for Wikipedia, are switched off.
Each token of the final revision is attributed to the editor of the revision WikiWho gives as its origin; tokens are
then mapped to the attribution's tokens (lower-cased alphanumeric) by sequence alignment."""
import re
from difflib import SequenceMatcher
import WikiWho.wikiwho as WW
from WikiWho.utils import iter_rev_tokens
WW.CHANGE_PERCENTAGE = -1e9; WW.UNMATCHED_PARAGRAPH = 1e9; WW.TOKEN_DENSITY_LIMIT = 1e9; WW.TOKEN_LEN = 10 ** 9
TOK = re.compile(r"[a-z0-9]+")


def wikiwho_chatbot_marks(events, final_tokens):
    """events: time-ordered list of (kind, text) with kind in {'W', 'Q', 'R'}; the last 'W' event is the final text.
    Returns a list of booleans over final_tokens: True where WikiWho's origin revision is a chatbot reply."""
    revs, editor, doc = [], {}, ''
    for k, (kind, text) in enumerate(events):
        text = str(text) if text is not None else ''
        if kind == 'W': doc = text; body = doc
        else: body = (doc + '\n\n' + text) if doc else text
        rid = k + 1; editor[rid] = 'C' if kind == 'R' else 'W'
        revs.append({'revid': rid, '*': body, 'timestamp': str(k), 'userid': 1 if kind == 'R' else 2, 'user': kind})
    if events and events[-1][0] != 'W':
        rid = len(events) + 1; editor[rid] = 'W'; revs.append({'revid': rid, '*': doc, 'timestamp': str(rid), 'userid': 2, 'user': 'W'})
    ww = WW.Wikiwho('student'); ww.analyse_article(revs)
    last = ww.revisions[ww.ordered_revisions[-1]]
    wt, wc = [], []
    for w in iter_rev_tokens(last):
        for t in TOK.findall(w.value.lower()):
            wt.append(t); wc.append(editor.get(w.origin_rev_id) == 'C')
    marks = [False] * len(final_tokens)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, list(final_tokens), wt, autojunk=False).get_opcodes():
        if op == 'equal':
            for q in range(i2 - i1): marks[i1 + q] = wc[j1 + q]
    return marks, len(ww.ordered_revisions), len(revs)
