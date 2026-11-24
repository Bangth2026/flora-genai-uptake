# Analysis log entry R25

Written on 24 November 2026, after a review of the ninth version of the manuscript asked for a test of the identified range on CoAuthor, and after the analyses below had run. No rule, parameter or label changes.

## CoAuthor (src/r25_coauthor_range.py, src/r25_coauthor_range_posthoc.py)

The reading and labelling code of `src/r24_coauthor.py` is repeated unchanged; the outputs below are added. Sessions are resampled (2,000 resamples, seed 2501); the 1,445 sessions come from 63 writers whom the release does not identify, so these intervals are too narrow.

- A-11. Per session, the identified range for the share of model text and the suggestion-inserted share (the platform's record). Mean shares: inserted 27.7%, C 24.4%, E 0.8%, E-after 0.0% (in CoAuthor the context sent to the model is text already in the editor, so echo cannot enter after the reply), X 2.6%, X-with-C 0.4%, T 71.0%, S 1.2%. Share of sessions whose inserted share lies inside the range: under A1-A3, N, B, R (lower end C, upper end C + E-after + X-with-C) 29.1% [26.7, 31.4]; without B 49.6% [47.0, 52.2]; without B and A1 56.5% [54.0, 59.0]; without any assumption 92.9% [91.6, 94.1]. The inserted share lay above the upper end under B in 63.8% of sessions (median 3.7 points) and below the lower end in 7.1% (median 0.5 points). Sessions at or above half: inserted share 12.9%; range under all assumptions 9.1% to 9.5%; without B 9.1% to 11.6%; without B and A1 9.1% to 12.4%.
- A-12. X subtypes: X-with-C 14.8% of X; X inserted by an accepted suggestion with no chatbot-first token in the same insertion 77.4%; writer insertions of more than five tokens 7.0%; other 0.7%. Suggestion text labelled T: 1.1% of the mean document. Of the sessions above the upper end under B, 97.1% contain such X or T.

## Post hoc (A-13, written after A-11 and A-12 were seen)

- Coverage with a tolerance of one and two points: under B 39.2% and 50.9%; without B 63.6% and 77.5%; without B and A1 69.6% and 81.1%. Adding the text of accepted suggestions shorter than eight tokens (X inserted by a suggestion alone, and T inserted by a suggestion), which B and R exclude, to the upper end under B: 86.0% (93.4% within one point); sessions at or above half 9.1% to 13.1% against 12.9%.

Reading: in CoAuthor B and R fail for one specific reason. An accepted suggestion shorter than n = 8 tokens contains no complete 8-gram, so it enters the editor as unmatched model text that is not beside chatbot-first text (against B) or as a short insertion labelled T (against R). The range under B then falls below the record in most sessions, by a few points.

## NIRVANA

The authors of NIRVANA have not replied since the OSF project became private on 22 November 2026 and an access request through OSF was not granted. The aggregate results of R22 and R23 are reported in the Supplementary Material (Section B.3.5 of the tenth version) so that readers of the earlier version can see them; the files of releases v1.22.0 and v1.23.0 are unchanged.
