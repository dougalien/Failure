# AFP — AI Failure Points

Single-file Streamlit app (also runs as a CLI) that measures where an LLM fails on
tasks with **objective, code-computed ground truth**, whether a **best-practice
prompt** fixes the failure, and how often the model is **confidently wrong**.

## Setup (Windows, PowerShell)

1. Install packages:
   ```powershell
   cd "C:\Users\allen\OneDrive\Documents\PythonCodes\AI Failur Points (AFP)"
   pip install -r requirements.txt
   ```
2. Add your Ollama key: copy `.streamlit\secrets.toml.example` to
   `.streamlit\secrets.toml` and paste the key:
   ```toml
   OLLAMA_API_KEY = "your-key"
   ```
3. Start the app:
   ```powershell
   streamlit run afp_app.py
   ```
   The sidebar defaults to **ollama**, host `https://ollama.com`, model `gemma4:31b`,
   and shows "Key source: Streamlit secrets" when the key is found.

Key lookup order: key typed in the sidebar → Streamlit secrets → environment variable.

Streamlit Community Cloud: paste the same `OLLAMA_API_KEY = "..."` line into the
app's **Settings → Secrets**. Run folders written there are not permanent, so use the
download buttons on the Results tab to keep your data.

CLI alternative (reads keys from environment variables only, e.g. `$env:OLLAMA_API_KEY="..."`):

```powershell
python afp_app.py --backend gemini --model gemini-2.5-flash --n 20
python afp_app.py --backend ollama --host https://ollama.com --model <model> --n 20
python afp_app.py --backend simulated --n 20      # FAKE data, pipeline test only
python afp_app.py --list-tasks
```

Each run writes a folder under `runs/` containing `raw.jsonl` (every prompt and full
response), `meta.json` (model, date, temperature, seed, prompts), summary CSVs and
three PNG figures.

## What is tested (12 tasks)

| id | Task | Failure it probes |
|---|---|---|
| letter_count | Letters in a word | character-level counting |
| list_count | Items in a list | counting repeats in a long list |
| grid_count | Symbols in a grid | 2-D counting (point-count analogue) |
| word_count | Words in a passage | counting words |
| multiplication | 4-digit × 4-digit | exact arithmetic |
| decimal_compare | 9.11 vs 9.9 type pairs (50% traps) | numeric magnitude |
| day_of_week | Weekday for a date 1800–2200 | calendar arithmetic |
| reversal | Reverse a string | character transformation |
| state_tracking | Objects through 8–14 swaps | multi-step state |
| exact_length | Sentence of exactly N words | hard output constraint |
| nonexistent_entity | Invented minerals, papers, effects, fossils | fabrication vs. abstaining |
| sycophancy | Correct answer, then user pushes a wrong number | caving to pressure |

## Design

* Same generated items go to both conditions (paired). Condition order is randomised.
* **Baseline** = plain question. **Best-practice** = accuracy-first system role,
  delimited task, task-specific step-by-step procedure, self-verification step,
  permission to answer UNKNOWN, request for calibrated confidence. Both share the
  identical `ANSWER:` / `CONFIDENCE:` format lines.
* No tools (code execution, search). A tools-enabled condition is an obvious next step.
* Groups are assigned **from the data**, not in advance:

| Group | Rule (defaults, adjustable) |
|---|---|
| No failure observed | baseline ≥ 95% |
| Prompt-fixable | significant gain and optimized ≥ 95% |
| Improved, not fixed | significant gain, optimized < 95% |
| Prompt-insensitive | no significant gain and gain < 10 pts |
| Inconclusive (increase n) | not significant, but gain ≥ 10 pts or optimized ≥ 95% |
| Prompt-harmed | significantly worse |

* Statistics: Wilson 95% intervals; exact McNemar test (paired); Fisher exact for the
  sycophancy task (items with a wrong first answer are excluded there).
* "Confidently wrong" = wrong with stated confidence ≥ 80, as % of scored answers.
* API errors are excluded and counted; missing `ANSWER:` lines are scored wrong but
  reported separately as parse failures.

## Caveats for any write-up

* n = 20 per condition only detects large effects. Use n ≥ 50 for claims.
* "Prompt-insensitive" means *not fixed by this prompt*, not *unfixable*.
* Verbalised confidence is a stated number, not an internal probability.
* Consequence and severity columns are illustrative and author-assigned.
* Hosted model versions change behind the same name; record run dates.
* Nonexistent-entity names are random syllables; spot-check `raw.jsonl` in case one
  matches a real entity.
* Scoring is regex-based. Spot-check a sample of `raw.jsonl` by hand before trusting
  any number, and report the spot-check rate.

## Background literature (verify before citing)

* Wei et al., 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.
* Kojima et al., 2022. Large Language Models are Zero-Shot Reasoners. NeurIPS.
* Kadavath et al., 2022. Language Models (Mostly) Know What They Know. arXiv:2207.05221.
* Xiong et al., 2024. Can LLMs Express Their Uncertainty? ICLR.
* Sharma et al., 2023. Towards Understanding Sycophancy in Language Models. arXiv:2310.13548.
* Wilson, 1927 (score interval); McNemar, 1947 (paired proportions test).
* Vendor prompt-engineering guides (Anthropic, OpenAI, Google) for the best-practice components.
