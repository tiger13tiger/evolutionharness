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
