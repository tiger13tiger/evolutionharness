# evolutionharness — Text-to-SQL Agent Eval

Evaluation harness for a text-to-SQL agent: golden dataset, guardrails, prompt-injection tests, CI gate.

```bash
python harness.py --repeats 5                    # mock agent, no API key needed
python harness.py --agent gemini --repeats 2     # free tier: GEMINI_API_KEY from Google AI Studio
python harness.py --agent claude --repeats 3     # Claude (ANTHROPIC_API_KEY)
```

Outputs `reports/report.md`, `summary.json`, `golden_results.csv`, `safety_results.csv`.
Exit code 1 when a metric misses `THRESHOLDS` in `harness.py`, so the GitHub Action blocks the PR.

| File | Purpose |
|---|---|
| `data/golden_set.jsonl` | Questions + gold SQL; `expect: answer/refuse` |
| `data/injection_set.jsonl` | Attack prompts (destructive SQL, override, PII, prompt leak) |
| `guardrails.py` | Read-only, single-statement, table allow-list, PII block |
| `agents.py` | `MockAgent` (seeded failure modes) and `ClaudeAgent` (tool-use loop) |
| `harness.py` | Metrics, repeated runs, report, CI gate |

## Prompt evolution

```bash
python evolver.py --agent mock --suite basic                          # offline demo
python evolver.py --agent gemini --generations 3 --candidates 2 --pace 4
```

Each generation: collect failed **train** cases → mutator LLM writes candidate prompts → accept only if
train improves, **holdout** (never shown to the mutator) does not drop, and safety stays 100%.
Candidates that drop safety rules or leak test questions/answers are rejected before evaluation.
In CI (`prompt-evolve` workflow) the winning prompt is opened as a PR to `prompts/system_prompt.txt`;
a human reviews the diff and report before merging.

`data/hard_set.jsonl` adds business-logic traps (data grain, ratios, period comparisons, launch windows, ambiguity).

## RAG few-shot retrieval

`--rag tfidf|fastembed` retrieves the top-k verified question→SQL examples from `data/example_bank.jsonl`
and appends them to the system prompt for each question. The bank is written to be distinct from the test set;
the harness refuses to run if any test question is a near-duplicate of a bank example (cosine ≥ 0.85).
`data/case_patterns.json` labels the analysis pattern each test case needs, so reports show **hit@k** and
task completion split by retrieval hit/miss — separating retrieval failures from generation failures.

## Results v1 (demo DB v1, DeepSeek `deepseek-chat`, 26 cases + 6 attacks, 3 runs each, Nov 2026)

| Config | All | Holdout | Hard cases | Exec. accuracy | Safety | Input tokens / q |
|---|---|---|---|---|---|---|
| A. Original prompt | 69.2% | 80.0% | 50.0% | 81.8% | 100% | 1,009 |
| B. Original prompt + RAG (bge-small) | 88.5% | 80.0% | 91.7% | 90.9% | 100% | 1,380 |
| C. Evolved prompt | 87.2% | 90.0% | 83.3% | 84.8% | 100% | 2,451 |
| **D. Evolved prompt + RAG (bge-small)** | **96.2%** | 90.0% | **100%** | **95.5%** | 100% | 2,702 |
| E. Evolved prompt + RAG (TF-IDF) | 92.3% | 96.7% | 94.4% | 90.9% | 100% | 2,675 |

- Prompt evolution and RAG fix *different* failures (evolution: data-grain rules, e.g. weekend vs weekday daily averages;
  RAG: worked patterns, e.g. ratios within a filtered brand, period growth), so they stack.
- RAG adds ~250–370 input tokens per question (+10%); prompt evolution added +143%.
- Several apparent failures were evaluator bugs found by inspecting real runs (extra columns, echoed years, ISO dates,
  rank numbers, window-function ORDER BY treated as result ordering). Each fix was checked against the mock agent's
  planted bugs so the evaluator did not become lenient.
- Known gap: G10 is ambiguous (combined vs per-game, all-time vs since launch) and should be rewritten.

## Test suite v2

Demo DB v2 (`demo_db.py`): 12 venues, 15 games, Jan–Jun 2026, plus realistic defects — unreported (NULL) revenue,
zero-play days, a venue closed mid-period, a refunds table, an un-cleaned raw feed (duplicates, negative plays,
future dates, unknown venues) and **planted indirect prompt injections** (a game name and a staff note carry instructions).

- 100 answerable/decline cases: `golden_set.jsonl` (14), `hard_set.jsonl` (12), `extended_set.jsonl` (74, source in
  `data/build_extended_set.py`). v2 rewrote the ambiguous G10 and made ratio questions exclude unreported revenue.
- 30 attack cases (`injection_set.jsonl`): direct and indirect injection, impersonation, UNION/alias PII tricks,
  comment tricks, PRAGMA/ATTACH/sqlite_master probes, encoded instructions, data tampering, secret and prompt leaks,
  resource exhaustion.
- New controls: 10 s query timeout and row cap; guardrails ignore text inside quoted literals.
- New metrics: `attack_attempt_rate` (runs where the agent tried something unsafe that controls stopped) and unsafe
  attempts during normal questions — shows whether an injection *fooled* the agent even when nothing bad happened.
