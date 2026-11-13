"""Incremental (online) version of the five-class attribution, standard library only.

Events of one student are fed in time order. At each editor snapshot, the n-grams that enter the editor at that
moment are labelled from the events seen so far, so a learning environment can react while the student writes:
  C  in an earlier reply, not first in a prompt, not shared
  E  first in a prompt and then in a reply (prompt-first echo)
  S  in a shared source
  T  entered in an edit of at most tau tokens
  X  entered in a larger edit, source not in the logs
An n-gram that entered before any reply contained it is relabelled E (editor-first echo) when such a reply arrives.
Cohort-level recurrence (the peer rule of the offline attribution) is not known while students write, so S uses the
shared texts given at start only.

    att = OnlineAttributor(shared=[...], n=8, tau=5)
    att.prompt(t, text); att.reply(t, text); new = att.snapshot(t, text)   # new: {ngram: label}
    tokens, classes = att.classify(final_text)
"""
from difflib import SequenceMatcher
from provenance import tokens, ngrams, ORDER


class OnlineAttributor:
    def __init__(self, shared=(), n=8, tau=5, delta=0):
        self.n, self.tau, self.delta = n, tau, delta
        self.shared = {g for s in shared for g in ngrams(tokens(s), n)}
        self.t_bot, self.t_pr = {}, {}
        self.entry = {}          # n-gram -> (time, inserted tokens)
        self.label = {}          # n-gram -> current label
        self.changed = set()     # n-grams whose label changed after entry
        self.prev = []

    def prompt(self, t, text):
        for g in ngrams(tokens(text), self.n):
            self.t_pr.setdefault(g, t)

    def reply(self, t, text):
        for g in set(ngrams(tokens(text), self.n)):
            if g in self.t_bot:
                continue
            self.t_bot[g] = t
            # editor-first echo: the n-gram was already in the editor when the chatbot first produced it
            if g in self.entry and self.entry[g][0] + self.delta < t and self.label[g] in "TXS":
                self.label[g] = "E"
                self.changed.add(g)

    def _decide(self, g, t, ins):
        bot = g in self.t_bot
        prompt_first = bot and g in self.t_pr and self.t_pr[g] < self.t_bot[g]
        if bot and not prompt_first and g not in self.shared and self.t_bot[g] <= t + self.delta:
            return "C"
        if bot and prompt_first:
            return "E"
        if g in self.shared:
            return "S"
        return "T" if ins <= self.tau else "X"

    def snapshot(self, t, text):
        cur = tokens(text)
        prev, n = self.prev, self.n
        # changed region: common prefix and suffix are skipped
        m = min(len(prev), len(cur))
        pre = 0
        while pre < m and prev[pre] == cur[pre]:
            pre += 1
        suf = 0
        while suf < m - pre and prev[len(prev) - 1 - suf] == cur[len(cur) - 1 - suf]:
            suf += 1
        a, b = prev[pre:len(prev) - suf], cur[pre:len(cur) - suf]
        new = {}
        if b:
            if not a or len(b) == 1:
                ins = len(b)
            else:
                ins = sum(j2 - j1 for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                          if op in ("insert", "replace"))
            lo, hi = max(0, pre - n + 1), min(len(cur) - n + 1, len(cur) - suf)
            for s in range(lo, max(lo, hi)):
                g = tuple(cur[s:s + n])
                if g not in self.entry:
                    self.entry[g] = (t, ins)
                    self.label[g] = new[g] = self._decide(g, t, ins)
        self.prev = cur
        return new

    def classify(self, final):
        A = tokens(final)
        rank = {c: i for i, c in enumerate(ORDER)}
        best = [len(ORDER)] * len(A)
        for i, g in enumerate(ngrams(A, self.n)):
            r = rank[self.label.get(g, "X")]
            for j in range(i, i + self.n):
                best[j] = min(best[j], r)
        return A, [ORDER[b] if b < len(ORDER) else "X" for b in best]
