"""Climate Data Assistant: chat with NASA/NOAA global temperature data.

A Mistral model (function calling) writes SQL, this code runs it read-only on
DuckDB, and a deterministic number check verifies that every decimal in the
answer traces back to a query result, the user, or an earlier verified answer.

Run:   MISTRAL_API_KEY=... python app.py
Colab: GRADIO_SHARE=1 python app.py   (prints a temporary public link)
"""
import os, re, json, time

import duckdb
import gradio as gr
import pandas as pd
from mistralai.client import Mistral

MODEL = "ministral-8b-2512"  # pinned: aliases can silently redirect
DATA_URL = "https://datahub.io/core/global-temp/_r/-/data/"

if not os.environ.get("MISTRAL_API_KEY"):
    raise SystemExit("Set the MISTRAL_API_KEY environment variable first.")
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# ---------------------------------------------------------------- data
annual = pd.read_csv(DATA_URL + "annual.csv")
monthly = pd.read_csv(DATA_URL + "monthly.csv")
monthly["Year"] = monthly["Year"].astype(str)  # "YYYY-MM" stored as text

con = duckdb.connect()
con.execute("""
CREATE OR REPLACE TABLE monthly_anomaly AS
SELECT
  Source AS source,
  CAST(strptime(Year, '%Y-%m') AS DATE) AS month_date,
  CAST(substr(Year, 1, 4) AS INTEGER) AS year,
  CAST(substr(Year, 6, 2) AS INTEGER) AS month,
  CAST(substr(Year, 1, 4) AS INTEGER) // 10 * 10 AS decade,
  Mean AS anomaly_c
FROM monthly
""")
con.execute("""
CREATE OR REPLACE TABLE annual_anomaly AS
SELECT
  a.Source AS source,
  CAST(a.Year AS INTEGER) AS year,
  CAST(a.Year AS INTEGER) // 10 * 10 AS decade,
  a.Mean AS anomaly_c,
  COALESCE(m.n, 0) AS months_of_data,
  COALESCE(m.n, 0) = 12 AS is_complete_year
FROM annual a
LEFT JOIN (
  SELECT source, year, COUNT(*) AS n
  FROM monthly_anomaly GROUP BY source, year
) m ON m.source = a.Source AND m.year = CAST(a.Year AS INTEGER)
""")

# ------------------------------------------------------ system prompt
SCHEMA = """
You can query a DuckDB database with two tables of GLOBAL temperature
ANOMALIES. Anomalies are differences in degrees Celsius from a baseline
average. They are NOT absolute temperatures.

TABLE annual_anomaly  -- one row per source per year
  source           VARCHAR  'GISTEMP' or 'GCAG' (exact, uppercase)
  year             INTEGER  e.g. 2025
  decade           INTEGER  start year of decade, e.g. 1990 = 1990-1999
  anomaly_c        DOUBLE   annual mean anomaly in degrees C
  months_of_data   BIGINT   number of months behind the annual value
  is_complete_year BOOLEAN  false if the year has fewer than 12 months

TABLE monthly_anomaly  -- one row per source per month
  source      VARCHAR  'GISTEMP' or 'GCAG'
  month_date  DATE     first day of the month, e.g. 2025-01-01
  year        INTEGER
  month       INTEGER  1-12
  decade      INTEGER
  anomaly_c   DOUBLE   monthly mean anomaly in degrees C

SOURCES (two independent datasets, stored in the same tables):
  GISTEMP: NASA GISS. Baseline 1951-1980. Covers 1880-01 to 2025-12.
  GCAG:    NOAA Climate at a Glance. Baseline 20th-century average.
           Covers 1850-01 to 2026-07.

IMPORTANT DATA FACTS:
  - The two sources use different baselines, so their values for the
    same year differ (2025: GISTEMP 1.19, GCAG 1.05). Never average,
    sum or directly compare values across sources. Always filter on
    one source, or show both side by side.
  - Before 1880, only GCAG has data.
  - GCAG 2026 is a partial year (January-July only). Its annual value
    is not comparable with full years. Use is_complete_year = true for
    year-on-year comparisons.
  - The 2020s decade is incomplete (2020 onwards only). Say so when
    comparing decades.
  - The data contains no absolute temperatures, no regional or country
    data, and no forecasts.
"""

RULES = """
RULES:
- Every number in your answer must appear in a run_sql result from this
  conversation. If you did not query it, do not state it.
- If the user names no source, query both and show each value side by
  side with its baseline. Never average, sum or subtract across sources.
- If asked to combine sources, explain why not, then show each source's
  value side by side.
- If the data cannot answer (absolute temperatures, regions, countries),
  say so plainly and explain what the data contains. You may offer
  related anomaly values, but only after querying them.
- The data has no forecasts. For future years, say so; never predict.
- Partial years are not comparable with full years. Do not say which is
  warmer; explain why.
- If run_sql returns an error, fix the query and try again.
- Final answer: 1-3 plain-English sentences, 2 decimals, degrees C.
  Name the source and its baseline.
- Mention a caveat only if it applies to the rows you returned.
- Do all calculations (averages, differences, counts, rankings) in SQL.
  Never calculate yourself.
- Only call a year partial if a returned row has is_complete_year = false
"""
SYSTEM = SCHEMA + RULES

# --------------------------------------------------------------- tool
def run_sql(query: str) -> str:
    q = query.strip().rstrip(";")
    ok = re.match(r"(?is)^\s*(select|with)\b", q)
    if not ok or ";" in q:
        return json.dumps({"error": "Only one read-only SELECT query is allowed."})
    try:
        df = con.sql(q).df()
        return df.head(50).to_json(orient="records", date_format="iso")
    except Exception as e:
        return json.dumps({"error": str(e)})

tools = [{
  "type": "function",
  "function": {
    "name": "run_sql",
    "description": "Run one read-only DuckDB SELECT query on the climate tables. Returns rows as JSON, or an error.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string",
                  "description": "A single DuckDB SELECT query"}},
      "required": ["query"]}}}]

# ------------------------------------------------------- number check
NUM = re.compile(r"\d+\.\d+")

def collect(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(abs(obj))
    elif isinstance(obj, dict):
        for v in obj.values(): collect(v, out)
    elif isinstance(obj, list):
        for v in obj: collect(v, out)

FEEDBACK = "These numbers in your answer"

def unsupported(answer, messages):
    allowed = [float(x) for x in NUM.findall(SYSTEM)]
    for m in messages:
        if not isinstance(m, dict):
            continue  # model drafts in this call: never trusted
        c = m.get("content") or ""
        if m["role"] == "tool":
            try: collect(json.loads(c), allowed)
            except Exception: pass
        elif (m["role"] == "user" and not c.startswith(FEEDBACK)) or \
             (m["role"] == "assistant" and not c.startswith("⚠")):
            allowed += [float(x) for x in NUM.findall(c)]
    bad = []
    for s in NUM.findall(answer or ""):
        x = float(s)
        tol = 0.5 * 10 ** -len(s.split(".")[1]) + 1e-9
        if not any(abs(x - a) <= tol for a in allowed):
            bad.append(s)
    return bad

# ------------------------------------------------------ model + loop
def call(messages, model=MODEL, tries=4):
    for i in range(tries):
        try:
            return client.chat.complete(model=model,
                messages=messages, tools=tools, tool_choice="auto")
        except Exception as e:
            if "429" in str(e):
                time.sleep(2 ** i * 15)
            else:
                raise
    raise RuntimeError("Still rate limited")

def chat(question, max_steps=6, history=None):
    messages = [{"role": "system", "content": SYSTEM}, *(history or []),
                {"role": "user", "content": question}]
    retried = False
    for _ in range(max_steps):
        msg = call(messages).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            bad = unsupported(msg.content, messages)
            if not bad:
                return msg.content, messages
            print("CHECK FAILED:", bad)
            if retried:
                return f"⚠ Unverified numbers {bad}: " + msg.content, messages
            retried = True
            messages.append({"role": "user", "content":
                f"These numbers in your answer are not in any run_sql result: {bad}. "
                "Query them with run_sql or remove them, then answer again."})
            continue
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            print("SQL:", args["query"])
            messages.append({"role": "tool", "name": tc.function.name,
                "content": run_sql(args["query"]), "tool_call_id": tc.id})
    return "Stopped: too many steps.", messages

# ----------------------------------------------------------------- UI
MAX_TURNS = 3

def sql_used(messages):
    qs = []
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            a = tc.function.arguments
            a = json.loads(a) if isinstance(a, str) else a
            qs.append(a["query"].strip())
    return qs

def to_text(c):
    if isinstance(c, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")

def clean_history(history):
    out = [{"role": m["role"],
            "content": to_text(m["content"]).split("\n\n**SQL run:**")[0]}
           for m in history if m["role"] in ("user", "assistant")]
    out = out[-2 * MAX_TURNS:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out

def respond(message, history):
    hist = clean_history(history)
    print("history sent:", [(m["role"], m["content"][:40]) for m in hist])
    answer, messages = chat(message, history=hist)
    qs = sql_used(messages)
    if qs:
        answer += "\n\n**SQL run:**\n" + "\n".join(f"```sql\n{q}\n```" for q in qs)
    return answer

demo = gr.ChatInterface(fn=respond, title="Climate Data Assistant",
    description="Ask about global temperature anomalies (NASA GISTEMP, "
                "NOAA GCAG). Answers can take 1–2 min on the free tier.",
    examples=["How hot was the Earth in 2010?",
              "What was the anomaly in March?"])

if __name__ == "__main__":
    demo.launch(show_error=True, share=os.environ.get("GRADIO_SHARE") == "1")
