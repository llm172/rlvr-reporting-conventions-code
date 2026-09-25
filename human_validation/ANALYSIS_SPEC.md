# Frozen analysis specification — 625 independent human judgments

Status: specified before the first independent formal human label. This document does not report human results. The 24 pilot/training responses are outside the formal sample and every statistical denominator below. Freeze this specification, the reader versions, sample, script and guidelines after the human pilot; do not select responses or modify readers after seeing formal labels.

## Inputs and estimand

`scripts/analyze.py` reads `coordinator/frozen_sample.json`, `population.json`, and `final_labels.json`. The frozen sample has exactly 625 distinct state/response keys: GSM8K Qwen1.5B initial 105, RL 61; Smol1.7B initial 106, RL 82; MATH500 shared initial 69 and trained seeds 83/84/85 with 62/70/70. The target finite populations are four GSM8K states of 1,319 responses each and four MATH500 states of 500 each. There is one MATH initial state, not three independent copies.

All reader outcomes are the previously frozen outcomes. The script neither reruns a verifier nor infers mathematical equivalence from strings. Independently human-assessed `strict_fidelity`, `fallback_fidelity`, `mv_fidelity` and `A_B_answer_equivalence` are required. Opaque identical committed-answer strings may be marked equivalent by the workflow; nonidentical answers require a human equivalence judgment. Anonymous candidate-slot fidelity judgments can be reconciled into these fields only after the intent locks.

The estimand is correctness of the answer the response actually commits to. It is not logical validity of the reasoning chain, latent capability, correctness somewhere in the response, or an identified causal contribution of formatting to learning. Checkpoints, generated response caches and the three MATH training seeds are fixed. These confidence intervals exclude uncertainty from new test questions, fresh model generations, new training runs and systematic human judgment error. Generalization to any of those populations requires a separately specified analysis.

## Sampling and weights

Within each state, strata are correctness-bit vectors: strict/fallback/MV for GSM8K and strict/MV for MATH500. Every nonempty stratum has positive inclusion probability. Within-stratum selection is simple random sampling without replacement, frozen independently of human labels. For stratum h, `N_h` is the original population count and `n_h` the sampled count; `pi_h=n_h/N_h`, unit weight `w_h=N_h/n_h`, and population fraction `W_h=N_h/N`.

For any binary outcome z, its estimate is `sum_h W_h mean_sample_h(z)`, equivalently `sum_sample w_i z_i/N`. All reported population metrics use these weights. The script verifies sample counts, unique keys, probabilities, weights, bit labels and full-cache reader accuracy. The unweighted A/B sample raw agreement is explicitly diagnostic and is never substituted for a population agreement rate.

## Human correctness and unresolved outcomes

Define H=1 for `CLEAR_ANSWER + CORRECT`; H=0 for `CLEAR_ANSWER + INCORRECT` or `NO_ANSWER`. `AMBIGUOUS`, `UNRESOLVED_CONFLICT`, and clear answers whose correctness remains `UNRESOLVED` are unknown. A no-answer label must be `INCORRECT`; ambiguous/conflicted commitments must remain `UNRESOLVED`. Suspected reference errors remain recorded and require human resolution rather than forced disagreement with the reference.

Compute binary lower and upper outcomes `H_L=1{H=1}` and `H_U=1{H!=0}`. Report `mean(H_L)` as the definitely-correct mass, the identification range `[mean(H_L),mean(H_U)]`, and the unresolved mass `mean(H_U-H_L)`. A single human-accuracy estimate is emitted only when the range collapses. Unknown outcomes are never silently removed or coded incorrect. Identification bounds are not confidence intervals; JSON/CSV keep them in separate fields.

## Per-state reader metrics and denominators

| Metric | Exact definition and denominator |
|---|---|
| Reader accuracy E | Full frozen-cache mean score, denominator all responses; no human sampling uncertainty |
| Human accuracy H | Weighted committed-answer correctness, denominator all responses; unresolved bounds retained |
| Joint FP | `P(E=1,H=0)`, denominator all responses |
| Joint FN | `P(E=0,H=1)`, denominator all responses |
| Net bias b | `FP-FN = mean(E)-mean(H)` |
| Conditional FPR | `P(E=1 | H=0) = joint_FP/P(H=0)`; reported separately |
| Conditional FNR | `P(E=0 | H=1) = joint_FN/P(H=1)`; reported separately |
| Extraction fidelity | Human-confirmed semantic faithfulness of the extracted candidate(s), conditional on `final_status=CLEAR_ANSWER` |
| Fidelity unresolved mass | `P(CLEAR_ANSWER and fidelity=UNRESOLVED)`, denominator all responses |
| Clear commitment / unresolved / reference issue | Separate weighted population rates |

The fidelity denominator includes every clear committed answer, including clear answers whose correctness is unresolved. Faithfulness concerns extraction of the complete committed answer (sign, units, tuple/set, multipart fields), not agreement of the reader score with the reference. For MV candidates the human examines the entire frozen extracted payload and applies the guidelines; the analyzer never selects a reference-matching candidate. `UNRESOLVED` fidelity gives lower/upper numerator contributions 0/1. A nonclear commitment is excluded from this conditional fidelity metric and remains visible in the all-response uncertainty and correctness metrics; its fidelity must be `NOT_APPLICABLE` or `UNRESOLVED`. MATH fallback is always inapplicable. A missing extraction for a clear answer requires a human fidelity judgment; it is not automatically considered faithful.

Unresolved H outcomes have joint FP range `[0,1]` only when E=1 and joint FN range `[0,1]` only when E=0. Because E is constant within a stratum, joint errors are obtained directly from the same human-bound stratum intervals. Bias is computed as E−H, preserving this identity. Conditional ratios use conservative numerator/denominator bounds; they are undefined when the estimated eligible denominator is zero and may have `[0,1]` uncertainty when a zero population denominator remains possible. Do not call joint FP/FN “FPR/FNR” without specifying these denominators.

## Gains, bias drift and fixed-seed contrasts

For each model and reader, `Delta_E=E_RL-E_initial`, `Delta_H=H_RL-H_initial`, and `Gap=Delta_E-Delta_H=b_RL-b_initial`. Positive Gap means the reader reports more improvement than the human semantic measure. Report joint-error components `FN_initial-FN_RL` (missed-correct recovery) and `FP_RL-FP_initial` (increased false credit). Their sum equals Gap for resolved labels. This is a measurement identity, not a causal mediation analysis.

For each MATH trained seed, subtract the shared initial H once. The fixed-seed mean contrast is `(H_83+H_84+H_85)/3 - H_initial`. The coefficient on the initial estimate and each of its interval endpoints is −1, not −1/3 and not three independent baseline contributions. Do not use the three seed comparisons as independent replications of the initial labels. Item overlap across states does not invalidate the simultaneous bounds; no independence across contrasts is assumed.

The primary family has three prespecified estimands:

1. Qwen1.5B GSM8K strict gain minus human gain.
2. Smol1.7B GSM8K strict gain minus human gain.
3. MATH500 mean human gain across the three fixed trained seeds, relative to one shared initial.

Other reader gaps and individual MATH seed contrasts are secondary, fully reported. The MATH primary estimand is listed on the strict row solely to avoid duplicate endpoint counting; H does not depend on reader. An average decrease does not establish that all three seeds decreased.

## Confidence intervals and simultaneous control

The analysis uses exact hypergeometric inversion within each stratum, not a normal approximation or ordinary bootstrap. For an observed count x in n sampled responses from N, retain all integer population success counts K for which both `P_K(X>=x)>=a/2` and `P_K(X<=x)>=a/2`. The returned interval is the smallest/largest retained K divided by N. Equal-tail inversion has at least `1-a` design coverage; discreteness makes it conservative. A census returns exactly `[x/N,x/N]`, because it has no sampling uncertainty. A noncensus stratum with zero observed errors retains a positive possible-error upper bound.

Before outcomes are aggregated, enumerate all declared binary stratum endpoints: H lower/upper, unresolved/clear commitment, status/correctness/raw agreement, A/B equivalence lower/upper, disagreement adjudication/review/reference-issue events, and each applicable reader's fidelity lower/upper/unresolved events. Let M be the number of these endpoint-by-stratum cells in noncensus strata. Set `a=0.05/M` (or 0.05 if M=0). Census cells do not consume error probability. The union bound gives simultaneous coverage of at least 95% for all these stratum quantities. This one family also covers every propagated primary and secondary contrast. No data-dependent endpoint selection and no additional nominal uncorrected significance tests are used.

Aggregate lower/upper confidence bounds with nonnegative population weights. For signed linear contrasts, use each lower endpoint for a positive coefficient and upper endpoint for a negative coefficient when calculating the contrast's lower bound, reversing for its upper bound. Human gains propagate the unknown-label worst cases at both states. Reader−human intervals reverse the human interval around the exact reader gain. Ratio intervals use conservative interval arithmetic and are clipped to [0,1]. These confidence bounds cover the entire unresolved-label identification interval; a printed confidence range is not merely the uncertainty of a midpoint imputation.

This common family is deliberately conservative and may yield wide bounds with 625 labels. The script reports the actual M and per-cell alpha. Wide intervals must remain wide; do not replace them with zero-width bootstrap intervals after observing few errors. Human disagreement, systematic common mistakes and training-run variation are not made small by these sampling calculations.

## Independent A/B agreement and adjudication

IAA uses the original locked A/B labels, before adjudication. Per state, report population-weighted status agreement, correctness agreement, answer-equivalence bounds and raw substantive agreement. Raw agreement requires equal status, correctness, self-correction/conflict/withdrawal/reference flags, anonymous fidelity-slot labels when present, and explicit `EQUIVALENT` commitments; differences in evidence-span wording or optional notes do not count as substantive disagreement. Answer equivalence is based on the separately recorded semantic judgment; an unresolved equivalence is preserved in its sensitivity range and does not count as raw agreement.

Report weighted nominal Cohen kappa for status and correctness, together with their weighted confusion tables. Kappa is a descriptive agreement statistic here, with no confidence interval claimed; it is undefined if both annotators put every weighted response in the same category. Free-form mathematical answers lack a fixed nominal category universe, so the script does not invent an answer-string kappa. Agreement intervals for the binary outcomes use the same simultaneous family as the main metrics. Kappa is not proof of label truth.

`adjudication_rate` denotes substantive A/B disagreements (`agreement_type != FULL_AGREEMENT`), while `reviewed_rate` uses the final `adjudicated` boolean to count actual third-review tasks, including the random agreement audit. The automatic reason `INDEPENDENT_A_B_AGREEMENT_NOT_SELECTED_FOR_AUDIT` is not a third review. The workflow additionally forces relevant reference issues and fidelity disagreements to human adjudication and audits approximately 10% of full agreements selected after equivalence resolution. Retain audit corrections and reasons. Do not report post-adjudication agreement as independent IAA.

## Prespecified claim decisions

Any Stage 2 anomaly/review note also forces third review, without counting a mere note difference as substantive A/B disagreement. Such review requests retain the locked Stage 1 answer and let the third person resolve any proposed correction.

**Case A, strong semantic support:** the strict-minus-human gain interval is wholly positive for the relevant model; the missed-correct recovery component is positive; and the false-credit increase is bounded sufficiently tightly. For the optional word “majority,” require the recovery lower bound divided by a positive fixed strict gain to exceed 0.5. Use 5 percentage points as the prespecified coarse upper tolerance for FP increase; disclose the actual bound. A claim about only a 4–5 pp residual requires a tighter ±2 pp gap interval and cannot be justified by the coarse tolerance. Describe support only for datasets/models whose corrected intervals meet the rule. Never upgrade to whole-chain reasoning or a complete causal decomposition.

**Case B, partial support:** direction is compatible with the proposed interpretation but intervals cross zero, exceed the prespecified tolerance, or support occurs only in some states/models. Limit semantic claims to the supported scope, retain reader-dependent measured-progress claims elsewhere, and publish all bounds. Failure to reject zero does not establish reader unbiasedness.

**Case C, semantic explanation unsupported:** fallback/MV human-audited false credit from intermediate or conflicting answers is large enough to account for the apparent recovery, or strict false-credit growth rather than recovered committed answers explains the gain. Withdraw the “primarily recovered missed correct answers” wording when its prerequisite interval fails or human evidence contradicts it. Retain independently supported claims about learned reporting conventions and reader-dependent measured gains. Report the observed decomposition and intervals; the script does not automatically assign a rhetorical case from arbitrary point estimates.

For MATH, a wholly negative mean-H interval supports a decline in fixed-seed mean committed-answer accuracy. Claim every individual seed declined only if every relevant simultaneous seed interval is negative. Reader choice may reverse measured progress even if this semantic endpoint is unresolved. No result permits post-label resampling, reader changes, endpoint replacement or selective seed omission.

## Completion gate and reproducible outputs

Run `python scripts/analyze.py --package .` from this package after the human workflow is complete. Python 3.10+ with the standard library is sufficient. The production gate requires all 625 final labels, independent A/B stage locks, a real completed human pilot followed by `protocol_lock.json`, and `completion_receipts.json`. It verifies recorded SHA256 values for the sample, population, guidelines, this analysis specification, analysis/workflow scripts, protocol, A/B lock files and final labels. It also checks that final A/B fields exactly preserve locked judgments, original text and candidates match their frozen sources, the two human signers differ, and each Stage 2 lock follows both Stage 1 locks. Missing, duplicate, unknown or invalid labels, wrong weights, score/stratum mismatches and changed frozen artifacts stop the run. These receipts preserve an audit trail; no program can itself establish that a person performed the asserted work.

With no final human labels the script writes only `PENDING_HUMAN_LABELS` metadata, empty CSV headers, and `XX.X [XX.X, XX.X]` LaTeX placeholders. It never substitutes AI labels. A complete authorized run writes `results/human_validation_results.json`, `metrics.csv`, `gains.csv`, `agreement.csv`, `paper_human_validation_table.tex`, and `appendix_human_validation_errors.tex`. JSON/CSV values are probabilities or probability differences; LaTeX expresses gains in percentage points. The compact main table contains dataset/model, reader, measured gain, human gain and gap. FP/FN/net-bias details are in the appendix table; gain decomposition and conditional diagnostics are in the accompanying data. The caption explicitly identifies stratified population weighting, independent double-blind annotation plus adjudication, and the committed-answer scope.

`python scripts/analyze.py --self-test` runs explicitly synthetic unit and end-to-end tests without creating synthetic production outputs. Temporary disk fixtures live only in the operating system temporary directory. Tests cover unequal weights, exact finite-population coverage, zero errors, censuses, unknown outcomes, no-answer semantics, shared MATH initial coefficients, human-only fidelity inputs, label integrity, false agreement and receipt blocking. End-to-end fixtures exercise 24 separate synthetic pilot examples, pilot locks and timing feedback, protocol freeze, all formal A/B stage locks, 10% agreement auditing, finalization, analysis acceptance and adversarial reference/lock checks. Synthetic tests and their fake attestations are never part of the human evidence.
