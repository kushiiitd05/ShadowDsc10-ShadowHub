# Prompt to feed to an LLM: write the improvement-plan report

Attach or paste these files with the prompt (they are the only source of truth; do not use outside knowledge for any number):
1. `RESULTS_AND_VERDICTS_2026-11-26.md` (every experiment, score and verdict, the 20-config search table)
2. `ENDTERM_AND_IMPROVEMENT_PLAN_2026-11-25.md` (sections 10–34: how each result was obtained)
3. `PLAN_V4_FINAL_2026-11-25.md` (baselines, gate rule, calendar)

---

You are a careful data-science reviewer writing for a university course (IIIT-Delhi CSE558, group DSC 10). Using only the attached files, write a report on what the improvement plan found for a 42-day, 1,115-hub daily order-forecasting problem scored by RMSLE.

**Rules you must follow**
- Every number must come from the attached files; say "not measured" when a number is missing. Do not invent results.
- Do not compare our scores with any outside leaderboard, competition winner or public score, and do not claim a rank. Our numbers are RMSLE on our own time-ordered dev folds and are not comparable to anything outside this project.
- Keep the gate rule exactly as written: mean gain ≥ 0.0024 (twice the measured noise 0.0012), on at least 2 of 3 dev folds (2–4), same sign across seeds 20260807/08/09, still true excluding open-but-zero rows, and fold 5 (summer 2014, season-matched to the test window) checked. Anything that misses the gate but is better on most folds is a "challenger", not a winner.
- State plainly when a result is the best of many tries on the same folds (selection bias), and when confirmation is still running.
- Do not mention server names, addresses, user names or key paths.

**Write these sections**
1. Problem, data, split and metric (one paragraph, no jargon beyond what the files use).
2. Starting point and the final recipe: the old 3-model blend (0.1250), `cs02` (0.1190 single seed, 0.1173 over 3 seeds), the final-retrain setting, the fold-4 dry-run score, and the hyper-parameter-search candidate `hp_017` with its confirmation status.
3. What worked, ranked by gain: feature fraction 0.2 with bagging (0.1312 → 0.1190), seed averaging, per-tier member, `hp_017`, the TabPFN-TS blend challenger. For each: gain, folds where it helps, why it plausibly helps.
4. What did not work: a table of every refuted idea with its measured result and a one-line reason (use the verdict table in the results file; group by family: targets, models, features, origins, post-hoc corrections, diagnostics).
5. The 20-config search: a short table of the top 5 and bottom 3, the pattern (large trees plus strong L2 plus large min child win; small trees lose), and the caveat about picking the best of 19.
6. Risks and limits: fold-to-fold noise, fold 5 noise, CV-vs-test gap unexplained, winter fold 1 behaviour, holiday features cannot help a holiday-free test window, single-seed results, anything still running.
7. Recommendation: which recipe to freeze on 3 Dec and why (by CV only), what to test next if time remained, and the exact steps of the final retrain (saved models, SHA-256, sanity checks).
8. End-term connection: how the improvement work feeds the end-term (x* model choice, what the coreset/sketching results show, what is still to re-run).

**Format**: Markdown, about 1,500–2,000 words, tables for numbers, no filler, each claim followed by the section of the attached files it comes from (for example "plan §25"). End with a list of five questions an examiner could ask about the gate rule, selection bias or the fold design, each with a two-sentence answer taken from the files.
