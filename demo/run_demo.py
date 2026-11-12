"""End-to-end demonstration of the five-class attribution on synthetic logs (no study data needed).
Usage: python demo/run_demo.py            (prints each token's class, class shares, U* and segment uptake)
The synthetic logs in demo/synthetic_logs.json describe one invented student; times are minutes on one clock."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import attribute, shares, segment_uptake, editor_entry, ngrams, tokens

logs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_logs.json")))
n, tau = logs["n"], logs["tau"]
A, classes, label, u_star = attribute(logs["final"], logs["replies"], logs["turns"], logs["snapshots"], logs["shared"], n=n, tau=tau)

print("Token classes (C chatbot-first, E echo, S shared source, T typed, X untraced insertion):\n")
line, cur = [], None
for t, c in zip(A, classes):
    if c != cur and line:
        print(f"  [{cur}] {' '.join(line)}"); line = []
    cur = c; line.append(t)
print(f"  [{cur}] {' '.join(line)}\n")
print("Class shares:", {k: round(v, 3) for k, v in shares(classes).items()})
print(f"Verbatim uptake U* = {u_star:.3f}")
t_ed, _ = editor_entry(ngrams(A, n), logs["snapshots"], n)
segs = [tuple(s) for s in logs["segments"]]
print("Segment uptake (chatbot-first n-grams entering the editor during each segment):", segment_uptake(segs, logs["replies"], label, t_ed, n))
expected = logs.get("expected_shares")
if expected:
    got = {k: round(v, 3) for k, v in shares(classes).items()}
    assert got == expected, (got, expected)
    print("Check: class shares match the expected values stored with the synthetic logs.")
