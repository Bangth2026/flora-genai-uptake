"""Five-class provenance attribution (Algorithms 1, 2 and 4 of the manuscript), self-contained.

Input for one student (all times are numbers or datetimes on one clock):
  final        final text of the artefact
  replies      list of (time, text) chatbot replies
  turns        list of (time, text) student turns (prompts; include any code attached to a turn)
  snapshots    list of (time, text) editor snapshots in time order
  shared       list of texts available to all students (readings, template, n-grams common to the cohort)

Output: one class per token of the final text:
  C chatbot-first, E echo (entered through the student first), S shared source,
  T typed (entered in edits of at most tau tokens), X untraced insertion (larger edits, source not in the logs).
Verbatim uptake U* is the share of class C.

Only the Python standard library is used. The study code (src/) adds dataset readers, the chance baseline and
the validation analyses; the rules below are the same.
"""
import re
from difflib import SequenceMatcher

TOKEN = re.compile(r"[a-z0-9]+")
ORDER = "CESTX"


def tokens(text):
    """Lower-case alphanumeric tokens, as in the study (prose and code alike)."""
    return TOKEN.findall(str(text).lower())


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def first_times(items, n):
    """Algorithm 1, lines 1-2: earliest time at which each n-gram occurs in a list of (time, text)."""
    first = {}
    for t, text in sorted(items, key=lambda x: x[0]):
        for g in ngrams(tokens(text), n):
            first.setdefault(g, t)
    return first


def editor_entry(final_grams, snapshots, n):
    """Algorithm 2, lines 1-7: for each n-gram of the final text, the time of the first snapshot that contains it
    and the number of tokens inserted or replaced between the previous snapshot and that one."""
    remaining = set(final_grams)
    entry_time, inserted = {}, {}
    prev = []
    for t, text in snapshots:
        if not remaining:
            break
        cur = tokens(text)
        new = remaining & set(ngrams(cur, n))
        if new:
            ops = SequenceMatcher(None, prev, cur, autojunk=False).get_opcodes()
            ins = sum(j2 - j1 for op, i1, i2, j1, j2 in ops if op in ("insert", "replace"))
            for g in new:
                entry_time[g] = t
                inserted[g] = ins
            remaining -= new
        prev = cur
    return entry_time, inserted


def attribute(final, replies, turns, snapshots, shared=(), n=8, tau=5, delta=0):
    """Return (token list, class list, n-gram labels, U*)."""
    A = tokens(final)
    G = ngrams(A, n)
    t_bot = first_times(replies, n)
    t_pr = first_times(turns, n)
    shared_grams = {g for s in shared for g in ngrams(tokens(s), n)}
    t_ed, ins = editor_entry(G, snapshots, n)
    label = {}
    for g in set(G):
        in_reply = g in t_bot
        prompt_first = in_reply and g in t_pr and t_pr[g] < t_bot[g]                      # Alg. 1, line 4
        shared_g = g in shared_grams                                                       # Alg. 1, line 5
        if in_reply and not prompt_first and not shared_g and g in t_ed and t_bot[g] <= t_ed[g] + delta:
            label[g] = "C"                                                                 # Alg. 2, line 9
        elif in_reply and (prompt_first or (g in t_ed and t_ed[g] + delta < t_bot[g])):
            label[g] = "E"                                                                 # R1 / R2 (Alg. 1 line 5, Alg. 2 line 9)
        elif shared_g:
            label[g] = "S"
        elif g in t_ed:                                                                     # Alg. 2, line 11
            label[g] = "T" if ins[g] <= tau else "X"
        else:
            label[g] = "X"
    # Alg. 2, line 12: token takes the highest-priority label among the n-grams covering it
    rank = {c: i for i, c in enumerate(ORDER)}
    best = [len(ORDER)] * len(A)
    for i, g in enumerate(G):
        for j in range(i, i + n):
            best[j] = min(best[j], rank[label[g]])
    classes = [ORDER[b] if b < len(ORDER) else "X" for b in best]
    u_star = classes.count("C") / len(classes) if classes else 0.0
    return A, classes, label, u_star


def segment_uptake(segments, replies, label, entry_time, n=8):
    """Algorithm 4: chatbot-first n-grams of each segment's replies that first entered the editor during the segment.
    segments: list of (start_time, end_time); replies: list of (time, text)."""
    out = []
    for start, end in segments:
        grams = {g for t, text in replies if start <= t < end for g in ngrams(tokens(text), n)}
        out.append(sum(1 for g in grams if label.get(g) == "C" and start <= entry_time.get(g, float("-inf")) < end))
    return out


def shares(classes):
    m = len(classes) or 1
    return {c: classes.count(c) / m for c in ORDER}
