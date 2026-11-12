"""Builds demo/synthetic_logs.json: one invented student whose work contains every provenance class.
Times are minutes on one clock. Typing is simulated one word per snapshot; pastes insert a whole block."""
import json, os

shared = ["learning analytics is the measurement collection analysis and reporting of data about learners and their contexts"]
own_draft = "my project examines how first year statistics students use worked examples during the weekly online quizzes"
reply_methods = ("we will collect quiz logs and short interviews from two tutorial groups and compare the time spent "
                 "on worked examples with quiz scores across the semester")
outside = ("the faculty survey from last year reported that most students open the examples only after a failed attempt "
           "which suggests that timing matters more than access")
typed_title = "a research proposal on the use of worked examples in first year statistics"

replies = [
    [6, "Here is a revised version: " + own_draft + " and how this relates to their final exam performance."],
    [8, "A possible method: " + reply_methods + ". You could also add a pilot week."],
]
turns = [
    [5, "Can you improve this sentence? " + own_draft],
    [7.5, "What methods could I use?"],
]

snapshots, doc, t = [], [], 0.0
def snap():
    snapshots.append([round(t, 3), " ".join(doc)])
for wd in typed_title.split():                 # T: typed word by word
    t += 0.05; doc.append(wd); snap()
for wd in shared[0].split():                   # S: definition from a shared reading, typed
    t += 0.05; doc.append(wd); snap()
t = 7.0; doc += own_draft.split(); snap()      # E: student's own draft, pasted after it went to the chatbot
t = 9.0; doc += reply_methods.split(); snap()  # C: pasted from the chatbot reply
t = 10.0; doc += outside.split(); snap()       # X: large insertion from a source the logs do not record

logs = dict(n=8, tau=5, final=" ".join(doc), replies=replies, turns=turns, snapshots=snapshots, shared=shared,
            segments=[[5, 8], [8, 12]],
            expected_shares={"C": 0.268, "E": 0.165, "S": 0.165, "T": 0.134, "X": 0.268})
here = os.path.dirname(os.path.abspath(__file__))
json.dump(logs, open(os.path.join(here, "synthetic_logs.json"), "w"), indent=1)
print("wrote synthetic_logs.json with", len(snapshots), "snapshots")
