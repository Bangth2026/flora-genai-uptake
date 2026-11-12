# Demo: five-class provenance attribution without the study data

`provenance.py` is a self-contained implementation (standard library only) of Algorithms 1, 2 and 4 of the manuscript. It follows the rules used in `src/`: an n-gram is chatbot-first (C) if it was in a reply no later than its first appearance in the editor and did not enter through the student first; echo (E) if a student turn or the editor had it before the first reply that contains it; shared source (S) if it occurs in a shared text; otherwise typed (T) or untraced insertion (X) by the number of tokens inserted at its entry. Each token takes the highest-priority label among the n-grams covering it (C > E > S > T > X).

```bash
python demo/make_synthetic.py   # writes demo/synthetic_logs.json (one invented student)
python demo/run_demo.py         # prints each token's class, class shares, U* and segment uptake
```

`demo.ipynb` runs the same steps and shows how n, τ and the direction rule change the classes.

Expected output of `run_demo.py`: C 0.268, E 0.165, S 0.165, T 0.134, X 0.268; segment uptake [0, 19].

To apply the method to other logs, supply for each student: the final text; chatbot replies and student turns with times; editor snapshots with times; and any texts shared by all students. Minimum data conditions are listed in the manuscript (Table 3 of the revised version): without editor snapshots only C⁻, E (prompt-first) and S can be computed; without clipboard events the classes are unchanged but cannot be validated against pastes. Choose τ from the distribution of tokens inserted per snapshot transition in your own logs.

The demo reproduces no result of the study.
