# AFP — AI Failure Points (v0.3.2)

Single-file Streamlit app (also a command-line tool) that measures where an LLM fails on
tasks with **machine-checkable answers**, what kind of prompting fixes the failure, and
how often the model is **confidently wrong**.

Version 0.2 replaces 0.1 after the problems recorded in `AFP_failure_log.md` (F-001).
The 0.1 code is kept unchanged in `legacy/afp_app_v0_1.py`. The n=100 data in `n=100/`
was made with 0.1, and can only be analysed with that version.

## What changed from 0.1

| Problem in 0.1 | 0.2 |
|---|---|
| Some answers scored by keyword matching or "first number in the text" | One required answer format per task, stated identically in every prompt; exact matching only; anything else is a **format failure**, reported separately |
| "UNKNOWN" offered only in the optimized prompt | `ANSWER: UNKNOWN` allowed in every condition, reported as an abstention |
| Nonexistent-entity task had no controls | Mineral task: half real minerals (known crystal system), half invented names |
| "Optimized" prompt included the solution method | Three conditions: **baseline**, **generic best practice** (no task help), **generic + task method** |
| Tasks too easy (7 of 11 at ceiling) | Difficulty levels 1–3 per task, up to 9-digit multiplication, 300-item lists, 80 swaps, 50,000-day date arithmetic |
| Pushback test: one mild round | Up to 3 rounds, adding claimed authority and social proof |
| Scores stored once | Every reply is **re-scored from its raw text** on load and compared with the stored score |
| Items generated in memory | Every item and its truth is written to `items.jsonl` **before** any model call |

The Ollama, Gemini and OpenAI calling code is unchanged from 0.1. In 0.3 the Anthropic call also records token usage and stop reason and can send an effort setting; in 0.3.2 it is streamed (see Money safeguards).

## Setup (Windows, PowerShell)

1. `pip install -r requirements.txt`
2. Key: copy `.streamlit\secrets.toml.example` to `.streamlit\secrets.toml` and set
   `OLLAMA_API_KEY = "your-key"` (or paste it in the sidebar).
3. `streamlit run afp_app.py`

Command line (reads keys from environment variables):

```powershell
python afp_app.py --backend ollama --levels 3 --n 5 --tasks letter_count,mineral_identity
python afp_app.py --list-tasks
python afp_app.py --backend simulated --n 5 --levels 1,2,3     # FAKE data, pipeline test
python afp_app.py --combine "runs\run1" "runs\run2"
python afp_app.py --backend ollama --resume "runs\<run folder>"
```

## Claude (Anthropic) runs and cost control (added in 0.3)

1. Put `ANTHROPIC_API_KEY = "sk-ant-..."` in `.streamlit\secrets.toml` (or paste it in
   the sidebar). Keys come from the Claude Console (console.anthropic.com); API use is
   billed separately from a Claude subscription.
2. Sidebar: Backend **anthropic**, pick a model. Prices (USD per million tokens) are
   pre-filled from the table in the code (source and date shown); edit them if they have
   changed.
3. **Max output tokens per reply** (default 32,000). Current Claude models think before
   answering, and thinking counts toward this limit and is billed as output. A reply
   that hits the limit is marked **truncated** and is not scored.
4. **Effort** (optional): sent as `output_config.effort`; the model default is used if
   left blank. It changes cost and results and is recorded in `meta.json`.
5. Run tab → **Estimate cost**. The estimate is arithmetic only:
   * input tokens: exact counts from Anthropic's free `count_tokens` endpoint for two
     items per task/level/condition, scaled to n (falls back to characters ÷ 3.5 and
     says so);
   * output tokens: **assumed** (1,500 / 3,000 / 5,000 per reply for baseline / generic /
     method) until you choose **Measured in run …**, which uses the average billed output
     tokens of an earlier run with the same model;
   * the pushback task is costed as if every round is used;
   * worst case = every reply uses the full max-token limit.
6. Tick "This run is estimated to cost $… Continue?". The Run button stays disabled
   until the estimate matches the current settings and is accepted.
7. **Budget stop**: after every call the app adds up the actual cost from the tokens
   the API reports, and stops the run when it reaches your budget (the call that crosses
   it is already paid for). Default budget = 1.5 × estimate.
8. Results tab → **Tokens and cost**: actual tokens, thinking tokens, truncations and
   USD per condition, next to the estimate made before the run.

Suggested first use: a pilot with the cheapest model (claude-haiku-4-5), n = 5,
2–3 tasks. Then estimate the full run with "Measured in run <pilot>".

CLI: `python afp_app.py --backend anthropic --model claude-haiku-4-5 --tasks letter_count --levels 3 --n 5 --estimate-only`
(drop `--estimate-only` to run; it asks for confirmation unless `--yes`; `--budget 2.50`
sets the stop; `--profile-from "runs\<pilot>"` uses measured output tokens.)

## Money safeguards (0.3.2)

* **One run, one session.** A running run holds a lock (`run.lock`, refreshed before every
  call). Resuming it from another window or session is refused while it may still be
  active, so the same call is not paid for twice. A lock older than 30 minutes is treated
  as left over from a crash.
* **Claude calls are streamed.** Long replies cannot time out and be retried. A request is
  retried only if it failed before the reply started (rate limit, overload, server error,
  connection refused); those are not billed. If a reply breaks off part-way, it is NOT
  retried; the tokens reported so far are counted as spent and the call is recorded as an
  API error. (The true billed amount for a broken-off reply may be a little higher than
  the tokens reported before the break.)
* **Spend of record = `raw.jsonl`.** Spend is recomputed from every saved call when a run
  starts or resumes, saved after every call, and shown on the Results tab. The budget stop
  uses this figure. A pushback-task item (up to 4 turns) is checked against the budget
  after the item, not after each turn.
* **Confirmation is used up by a run.** After each run, the estimate and the "Continue?"
  tick are cleared; a new run needs a new estimate and confirmation.
* **The estimate uses measured output tokens by default** when an earlier run with the same
  model exists (the assumption is only used when there is no measurement).
* **n below the minimum is flagged before running.** If n is below "Minimum scored items
  per condition", the Run tab says no groups will be assigned, and the results say why
  ("scored answers baseline 9, generic 9, method 9; minimum is 10").
* **Duplicate calls** (the same item and condition answered twice) are counted and shown.
  The **first** successful answer is scored; a later answer never replaces it.

## Before any large run

1. Run a **pilot**: n = 5, the tasks and levels you plan to use.
2. Download **Audit sample (CSV)** from the Results tab and check the scoring by hand.
3. Only then run the full n. Calls = task-levels × n × conditions (plus pushback turns).

## Tasks and levels

| id | Task | Levels | Required answer |
|---|---|---|---|
| letter_count | Letters in a string | L1: 12-letter pseudo-word; L2: 30-letter pseudo-word; L3: 60-letter pseudo-word | the number only, written with digits (for example 1234) |
| list_count | Items in a list | L1: 40 items; L2: 120 items; L3: 300 items | the number only, written with digits (for example 1234) |
| grid_count | Symbols in a grid | L1: 8 x 10 grid; L2: 15 x 20 grid; L3: 25 x 30 grid | the number only, written with digits (for example 1234) |
| word_count | Words in a passage | L1: 40 words; L2: 150 words; L3: 400 words | the number only, written with digits (for example 1234) |
| multiplication | Long multiplication | L1: 4-digit x 4-digit; L2: 6-digit x 6-digit; L3: 9-digit x 9-digit | the number only, written with digits (for example 1234) |
| number_sort | Sort decimal numbers | L1: 5 numbers, 1-2 decimals; L2: 10 numbers incl. negatives, 1-3 decimals; L3: 20 numbers incl. negatives, 1-4 decimals | the numbers in ascending order, separated by a comma and a space, each written exactly as in the question (for example -2.5, 3.14, 3.9) |
| day_of_week | Day of week for a date | L1: years 1950-2050; L2: years 1800-2200; L3: years 1583-4000 | the weekday name in English, for example Monday |
| date_arithmetic | Add days to a date | L1: 10-100 days; L2: 100-2,000 days; L3: 2,000-50,000 days | the date in the form YYYY-MM-DD |
| reversal | Reverse a string | L1: 12 letters; L2: 30 letters; L3: 60 letters | the reversed string only, lowercase, no spaces |
| state_tracking | Track objects through swaps | L1: 5 people, 10 swaps; L2: 7 people, 30 swaps; L3: 10 people, 80 swaps | one person's name only |
| exact_length | Text of exactly N words | L1: 12-20 words; L2: 40-60 words; L3: 100-150 words | the text itself on this single line; words separated by single spaces; no digits, hyphens or apostrophes |
| mineral_identity | Real vs invented minerals | L1: 50% real minerals, 50% invented names | one of: cubic, tetragonal, orthorhombic, hexagonal, trigonal, monoclinic, triclinic |
| sycophancy | Keep correct answer under pushback | L1: 2x2-digit product, 1 round of pushback; L2: 3x2-digit product, 2 rounds (adds claimed authority); L3: 3x3-digit product, 3 rounds (adds social proof) | the number only, written with digits (for example 1234) |

The Method tab in the app lists every scoring rule. The "Prompts & items" tab shows the
exact text sent to the model for any task and level. Nothing else is sent: no history and
no hidden instructions.

## Groups (from the data, per task and level)

| Group | Study group | Rule |
|---|---|---|
| Reliable without guidance | — | baseline ≥ threshold (default 95%) |
| Fixed by generic prompting | Prompt-fixable | generic significantly better and ≥ threshold |
| Fixed only with task method | Prompt-fixable (needs the method) | method significantly better and ≥ threshold |
| Improved, not fixed | Prompt-improved, not fixed | significantly better, below threshold |
| Not fixed by prompting | Prompt-insensitive | no significant gain, best gain < margin |
| Prompt-harmed | — | significantly worse |
| Inconclusive | — | not significant, but reaches threshold or gains ≥ margin |

Exact McNemar tests (baseline vs generic, baseline vs method), Holm-corrected; Fisher exact
for the pushback task; Wilson 95% intervals.

## Run folder contents

`meta.json` (settings, prompts, versions) · `items.jsonl` (all items and truths) ·
`raw.jsonl` (every request and full reply) · `summary*.csv` · `outcomes.csv` ·
`mineral_controls.csv` · `calibration.csv` · `confidently_wrong.csv` ·
`audit_sample.csv` · three PNG figures. Keep each run's **Whole run (.zip)**.

## Interrupted runs, limits, combining

* A run stops after 5 API errors in a row (usually a usage limit). Resume it later:
  finished calls are skipped.
* Combining requires the same backend, model, temperature, seed, prompts and app version.
* On Streamlit Community Cloud, saved runs disappear on restart: keep the zips.

## Limits to state in any write-up

* Results hold for the model, levels and prompts tested; "Not fixed" means not fixed by
  these prompts at this n.
* Model replies are not guaranteed identical between runs even at temperature 0. Items,
  prompts and scoring are reproducible; replies are recorded.
* Stated confidence is a number the model writes, not an internal probability.
* Real-mineral list: well-established species with one crystal system; trigonal species
  are excluded (trigonal vs hexagonal naming). Five were spot-checked against Wikipedia.
  Doug to review. Invented names are random and checked against a list of real names;
  spot-check `items.jsonl`.
* Consequence and severity columns are illustrative and author-assigned.

## Background literature (verify before citing)

* Wei et al., 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.
* Kojima et al., 2022. Large Language Models are Zero-Shot Reasoners. NeurIPS.
* Kadavath et al., 2022. Language Models (Mostly) Know What They Know. arXiv:2207.05221.
* Xiong et al., 2024. Can LLMs Express Their Uncertainty? ICLR.
* Sharma et al., 2023. Towards Understanding Sycophancy in Language Models. arXiv:2310.13548.
* Wilson, 1927 (score interval); McNemar, 1947 (paired proportions); Holm, 1979 (multiple tests).
