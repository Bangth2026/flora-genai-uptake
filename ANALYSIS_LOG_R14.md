# Analysis log entry R14

Written after all earlier results (S1 to R13) were known and after its analyses had run. All analyses are exploratory. No measure, class definition, parameter value or code of the attribution changes; every earlier result is unchanged.

## B. Benchmark (src/r14_benchmark.py; same simulated students and scores as R11 to R13)

B-16. AURC difference between the attribution and earliest plausible attribution with abstention by entry size, with routes left out singly, with the shared-source and streaming routes left out together, and with each further route left out in addition.

Result, 20 routes: all routes -0.0053; shared-source route left out -0.0037; streaming route left out -0.0034; both left out -0.0021. With both left out, leaving out one more route gives -0.0014 (C paste, C paraphrase 25%) to -0.0026 (X other outside text): the remainder is spread over the chatbot routes, none above 0.0007. Leaving out the outside-text route alone widens the difference to -0.0067: both methods leave all its tokens undecided, so removing them shortens the abstention tail of both curves and scales up the difference between them; it does not favour either method. 13 routes: all -0.0015; shared-source route left out -0.0005; C paraphrase 10% -0.0007; C paraphrase 25% -0.0008; every other route -0.0016 to -0.0022.

B-17. Snapshots thinned to 60 and 300 s, tau 5 and 20, n-gram and token entry rules, with the current code. The 300 s values equal those of the R10 run (C 0.933 / 0.847, E 0.933 / 0.792, S 0.923 / 0.971; n-gram rule T 0.137 / 0.003 and 0.265 / 0.047, X 0.707 / 0.991 and 0.709 / 0.961). Token rule: T 0.903 / 0.007 (tau 5) and 0.494 / 0.090 (tau 20); X 0.709 / 0.995 and 0.717 / 0.971; student-level MAE of U 0.057.

B-18. Tokens of the reading-highlight route by class at full resolution and with snapshots at least 5, 15, 60 and 300 s apart: 108 of 3,971 tokens are E at every resolution except 60 s, where they are S. These are highlights that a reply repeated after they entered the editor (R2 precedes R3). In the run thinned to 60 s, the first kept snapshot containing them falls after that reply, so R2 no longer holds and R3 gives S; at 5, 15 and 300 s a kept snapshot falls between entry and reply. The non-monotonic S recall in Table B7 is therefore a property of where kept snapshots fall, not an error in the table.
