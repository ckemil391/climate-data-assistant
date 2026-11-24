# Climate Data Assistant: chat with NASA/NOAA temperature data on the Mistral API

Ask a question in plain English ("How hot was the Earth in 2010?"). A Mistral model uses **function calling** to write SQL, my code runs it read-only on **DuckDB**, and the model answers with the right source, baseline and caveats. A **deterministic number check in code** verifies that every decimal in the answer comes from a query result before it reaches the user.

I built it as a portfolio project to learn how to build and evaluate an LLM application end to end, and to measure the result honestly instead of eyeballing it.

```
Question ─► Mistral (ministral-8b-2512) + system prompt (schema + rules)
               │ requests a tool call
               ▼
           run_sql(query) ── read-only check ──► DuckDB
               │ rows as JSON (or the error)
               ▼
           Mistral writes the answer ─► number check (code) ─► user
                                         └─ unsupported numbers → one retry with feedback,
                                            then the answer is flagged "⚠ Unverified"
```

**Stack:** Python · Mistral API (`mistralai` SDK, function calling) · DuckDB · pandas · Gradio

---

## Results

I wrote a **16-question test set before tuning anything**, with expected answers computed by my own SQL ([`eval/test_set_v2.json`](eval/test_set_v2.json)). It covers lookups, aggregation, ranking, a window function, counts and thresholds, plus traps that came out of profiling the data: a partial year, mixing two sources, absolute temperatures, regional data, forecasts, missing years, and an ambiguous source.

**Scoring (strict, the same for every run, scored by hand):** a pass needs the question's pass rule fully met, **every number taken from a SQL result**, and **no false statements**.

| Run | What changed | Score |
|---|---|---|
| Baseline | — | **9/16** |
| Round 2 | Bug fix (ISO dates in tool output) + clearer rules: numbers only from SQL, show both sources, no forecasts, caveats only if they apply | **10/16** |
| Round 3 | + "do all calculations in SQL" + **code-level number check** | **14/16** |
| Round 4 | + one rule targeting the partial-year confusion | **13/16** |

Each configuration ran **once**, so differences of ±1 are within run-to-run noise. The robust result: **invented numbers appeared in every run before the code check, and in none after it.**

*Spec change, documented:* after the baseline I decided that when no source is named, the answer should show both sources side by side. Previously it defaulted to GISTEMP. I updated Q16 and rescored the baseline, so every run is measured against the same spec. The run files still contain Q16's old pass rule; Q16 was scored by hand under the new one.

Raw answers, the SQL each run executed, and the expected results are in [`eval/`](eval/).

## What I learned

**A prompt rule asks; code enforces.** "Every number must come from a query" was in the system prompt from round 2, and the model still invented numbers. For "How hot was the Earth in 2010?" it produced a different wrong pair in each of three runs (0.63/0.62, 0.65/0.60, 0.62/0.65), all about 0.1 °C too low. A short check in code fixed it. After each answer, the check extracts every decimal and compares it against the numbers in that conversation's SQL results. An unsupported number triggers one retry with feedback. With the check, the same question was caught, the model queried, and the answer showed the real values: **0.72 / 0.67**.

**Push arithmetic into SQL.** The 8B model averaged rows in its head and got 0.41 when the answer is 0.39. Asking it to calculate in SQL makes every result auditable, and also checkable by the number check.

**Check what you feed the model before blaming it.** In the baseline the model said July when the answer was September. The cause was my tool: it serialised dates as epoch milliseconds. Fix: `date_format="iso"`.

**Separate systematic failures from random ones.** Q5 and Q10 failed in all four runs, both because the model confuses which year is partial (in Q5 it calls 2025 partial, when only GCAG's 2026 is). That's systematic and needs a root-cause fix. A targeted prompt rule in round 4 did *not* fix it. My hypothesis is that the schema description places 2025 example values next to the "2026 is partial" note; I haven't tested this yet. Q8 passed in round 3 and failed in round 4 ("most recent was 2024", actually 2025): a real number with the wrong label, which looks like random variation and is something the number check can't catch by design.

**Error recovery works when errors go back to the model.** `run_sql` returns database errors as JSON instead of raising them. In a fault-injection test I deliberately broke the first query (renamed a column). The model read DuckDB's error, fixed the SQL and answered correctly. In the reference cookbook ([`text_to_SQL.ipynb`](text_to_SQL.ipynb)) the same kind of error crashed the loop. Recovery also happened naturally during evaluation (Q11, Q16).

## How it works

**Data first.** I profiled the [DataHub `core/global-temp`](https://datahub.io/core/global-temp) data (323 annual rows, 3,871 monthly rows) before modelling it. That revealed two sources with **different baselines** (GISTEMP 1951–80, GCAG 20th-century average), different coverage, a **partial year** (GCAG 2026, January–July) and a text-encoded month column. Those findings became table design (`is_complete_year`, `decade`), facts in the schema description and trap questions in the test set.

| Table | Columns |
|---|---|
| `annual_anomaly` | source, year, decade, anomaly_c, months_of_data, is_complete_year |
| `monthly_anomaly` | source, month_date, year, month, decade, anomaly_c |

**The model never touches the database.** It returns a structured tool call; my code decides what actually runs. Guardrails at that boundary:
- `run_sql` accepts exactly one `SELECT`/`WITH` statement, caps the result at 50 rows, and returns errors instead of raising them
- the tool loop is bounded (`max_steps=6`)
- the model version is pinned (`ministral-8b-2512`), because an alias was silently redirected to a different model
- 429 rate-limit errors are retried with back-off

**The number check across a conversation.** The UI supports follow-up questions ("Was 2000 cooler?"). Earlier turns are resent to the model, because it has no memory between calls. That raised a design question: which numbers should a follow-up answer be allowed to repeat? Trust is transitive: a number is allowed if it comes from a query result, from the user, or from an earlier answer that itself passed the check. It's never allowed from a flagged answer, the model's unverified draft, or the check's own feedback message. Those rules are unit-tested without any API calls. Single-turn behaviour is unchanged, so the evaluation scores still apply.

**UI.** A Gradio chat shows the SQL that ran under every answer, so each number can be traced to its query.

## Run it

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY=...        # never commit the key
python app.py                     # opens a local Gradio app
```

In Google Colab: `!GRADIO_SHARE=1 python app.py` prints a temporary public link. On the Mistral free tier (2 requests/min when I tested), one answer can take 1–2 minutes.

## Repository

| File | What it is |
|---|---|
| `app.py` | The runnable app: data load, tables, tool, number check, chat loop, UI |
| `Climate_change.ipynb` | How it was built and measured: profiling, modelling, fault injection, evaluation runs |
| `text_to_SQL.ipynb` | Mistral's text-to-SQL cookbook run first to learn function calling (Chinook database) |
| `eval/test_set_v2.json` | 16 questions, pass rules, SQL-derived expected answers |
| `eval/run_*.json` | Answers and executed SQL for the baseline and rounds 2–4 |

## Limitations

- **Small model, small test.** An 8B model on the free tier, 16 questions, one run per configuration: enough to find failure patterns, not to claim a precise accuracy rate.
- **The number check** covers decimals only (not counts or years), ignores signs, and can't detect a real number with the wrong label. In multi-turn chat, trust is transitive, so these blind spots carry over to follow-up answers.
- **The read-only guard is a regex.** In production, a read-only database user would enforce it.
- **Unresolved:** the partial-year confusion (Q5/Q10). Testing the UI also found a gap the test set didn't cover: with no year given ("What was the anomaly in March?"), the model silently picks a year instead of asking or saying what it assumed. That becomes a test case for the next version.
- The app downloads the data from DataHub at startup.

## Next steps

- Fix the partial-year confusion at its source (the schema description) and re-evaluate.
- Run each configuration several times and add the ambiguous-period question to the test set.
- Compare against a larger Mistral model to quantify the cost/accuracy trade-off.
- Log every question, SQL, result and check failure, and track the pass rate on the test set after every prompt or model change.

## Taking this to a customer

The method carries over to a proof of value on a customer's own data:
1. **Agree on success criteria first.** Build the question set and what counts as correct *with* the customer, measure a baseline, then improve against it.
2. **Governance.** Read-only access, row limits, scoped schemas, and no sensitive data sent to external endpoints; consider a private deployment where needed.
3. **Trust.** Show the source and the SQL with every answer, and prefer "the data can't answer that" over a confident guess.
4. **Monitoring and cost.** Track the pass rate on a regression set after every change. Count the calls per question (here 2–5), since every safeguard, such as a retry, adds calls.

## Data and attribution

Global temperature anomalies from **NASA GISS (GISTEMP)** and **NOAA National Centers for Environmental Information (Climate at a Glance, GCAG)**, via [DataHub `core/global-temp`](https://datahub.io/core/global-temp). Anomalies are differences from each source's baseline, not absolute temperatures.
