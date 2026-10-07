# Data Analysis Agent — Backend

A Groq-powered agent that answers natural-language questions about any CSV by
calling deterministic analysis tools. **The LLM never does the maths**: it picks
tools, reads their results, and writes the answer. Pandas / NumPy / SciPy /
Matplotlib do all computation.

## Architecture

```
backend/
├── agent/
│   ├── orchestrator.py     # agent loop: LLM -> tools -> LLM ... -> answer
│   ├── executor.py         # parallel vs sequential execution of a turn's tool calls
│   ├── tool_registry.py    # native tool schemas, argument validation, safe dispatch
│   ├── llm_client.py       # Groq client + friendly error types
│   ├── prompts.py          # short system prompt (+ hook for future memory context)
│   ├── config.py           # paths + env-driven settings
│   ├── data_loader.py      # generic CSV loader with date auto-detection
│   └── tools/              # 39 tools, organised in 6 modules
│       ├── _base.py            # @tool decorator, registry, ToolError, schema builders
│       ├── _helpers.py         # shared filters, column lookup, JSON sanitising, time aggregation
│       ├── mathematical_operations.py  # mean median sum min max count std variance quantile group_aggregate
│       ├── data_manipulation.py        # filter_rows sort_rows sample_rows | fill_missing drop_missing
│       │                               #   remove_duplicates rename_columns convert_dtype select_columns
│       ├── data_summary.py             # dataset_profile statistical_summary missing_value_summary
│       │                               #   unique_value_summary value_counts distribution_info
│       ├── graphs.py                   # line_plot bar_plot scatter_plot histogram box_plot heatmap
│       ├── correlation_analysis.py     # correlation_matrix pairwise_correlation covariance
│       │                               #   linear_regression trend_analysis
│       └── outlier_analysis.py         # iqr_outliers zscore_outliers outlier_summary
├── api/                    # Flask: app.py, routes.py, dataset_store.py
├── tests/                  # 69 tests (no API key needed)
├── data/sample_sales.csv
├── run_api.py  cli.py  demo_tools_only.py
└── requirements.txt  .env.example
```

The modules are only for organisation. Groq sees **each function as its own tool**
through native function calling (`tools=[...]`), so there is no home-made JSON protocol.

## One tool interface

```python
@tool(name="group_aggregate", description="...", parameters=obj({...}, required=[...]),
      category="mathematical_operations", mutates=False)
def group_aggregate(ctx, group_by, agg, column=None, sort="desc", top_n=None, filters=None): ...
```
* Every tool returns a JSON dict; predictable problems raise `ToolError` and reach the
  model as `{"error": "..."}` so it can recover (e.g. "Column 'revnue' not found. Did you mean: revenue?").
* Analytical tools and all graphs take the **same optional `filters`**:
  `[{"column": "region", "operator": "==", "value": "West"}]`
  (`== != > >= < <= in not_in contains is_null not_null`; text matching is case-insensitive;
  numeric/date values sent as strings are coerced). Cleaning tools and `dataset_profile`
  deliberately don't take filters.
* Arguments are validated against each tool's schema before running (unknown tool/parameter,
  missing/wrong-typed values, enums, ranges). Only registered tools can execute; there is no
  code execution and no `df.query()` on model-written strings.
* Unexpected exceptions are logged server-side and reach the model/user as a generic message.

## How requests run

**Single-step**: "What's the average revenue?" -> `mean(column=revenue)` -> answer.

**Chained (dependent)**: "Find the region with the highest average revenue and plot its monthly trend."
1. turn 1: `group_aggregate(group_by=[region], agg=mean, column=revenue)`
2. the result goes back to Groq, which reads the top region (say `West`)
3. turn 2: `line_plot(x=order_date, y=revenue, resample=M, filters=[region == West])`
4. turn 3: final answer.

Each dependent step is a separate LLM turn, so it always sees the previous result.

**Parallel (independent)**: "Average revenue, median revenue, distinct customers and a revenue-over-time chart."
Groq emits all the calls in one turn; the executor runs them concurrently in a thread pool
and returns results in call order. Charts use Matplotlib's object API (no shared global state),
so concurrent charts are safe.

**Safety rule for parallelism**: the prompt tells the model to put only independent calls in one
turn. Additionally, tools that change the working data (`fill_missing`, `drop_missing`,
`remove_duplicates`, `rename_columns`, `convert_dtype`, `select_columns`) are barriers: they never
run alongside other calls, so ordering is always respected. Cleaning affects only the current
question; the stored dataset is never modified.

Guards: max 8 LLM turns and 24 tool calls per question (then Groq is asked for a final answer without
tools); malformed tool calls (`tool_use_failed`) are retried with a nudge; oversized tool results are cut.

## Setup & run

```bash
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then put your GROQ_API_KEY in .env
```

| Command | Needs key? | What it does |
|---|---|---|
| `python -m unittest discover -s tests -t . -v` | no | full test suite (fake LLM + real Groq SDK on a mock transport) |
| `python demo_tools_only.py` | no | runs tools directly, incl. a manual top-region -> trend chain |
| `python cli.py --data data/sample_sales.csv` | yes | terminal chat, prints each tool step |
| `python run_api.py` | yes | API on http://localhost:5000 (open `frontend/index.html`) |

### Environment variables
| Variable | Required | Default |
|---|---|---|
| `GROQ_API_KEY` | yes (server only) | none |
| `GROQ_MODEL` | no | `llama-3.3-70b-versatile` |
| `MAX_TOOL_ROUNDS` | no | 8 |
| `MAX_PARALLEL_TOOLS` | no | 4 |

## API

| Method | Endpoint | Notes |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/tools` | registered tools: name, category, description, parameters |
| POST | `/api/upload` | multipart `file` (.csv, up to 25 MB) -> `{dataset_id, name, schema}` |
| GET | `/api/datasets/<id>/schema` | dataset `sample` is preloaded |
| POST | `/api/ask` | `{dataset_id, question}` -> `{answer, tool_trace, plot_urls, rounds}` |
| GET | `/api/plots/<file>` | chart PNG |

`tool_trace` entries: `{round, tool, category, arguments, ok, error, summary, duration_ms, parallel}`
(`tool` and `arguments` are unchanged from the previous version, so older frontends still work).
Error responses are short JSON messages (no tracebacks): 400 bad input, 404 unknown dataset,
429 rate limit, 502 Groq problem, 500 setup/unexpected.

## Example questions
1. What columns does this dataset have, and is the data clean?
2. What is the average revenue by region?
3. Find the region with the highest average revenue and plot its monthly trend.
4. Give me the average revenue, median revenue and number of orders, and plot revenue over time.
5. Are there outliers in revenue? Show the biggest ones.
6. What is correlated with revenue? Show a heatmap.
7. Compare revenue distribution across categories with a box plot.
8. Which category has the most orders in the North region, and what is its total revenue?
9. Fill missing ratings with the median, then give the average rating by category.
10. Is revenue trending up or down each week for Electronics orders?

## Known limits
* Datasets live in server memory (lost on restart). Memory/RAG is intentionally not implemented; the
  `extra_context` argument of `run_agent` and `prompts.build_system_prompt` is where it would plug in.
* 39 tool schemas cost roughly 6k prompt tokens per LLM call. On Groq's free tier, multi-step questions can
  hit the tokens-per-minute limit; requests then fail with a friendly "rate-limited" message, so retry after a few seconds.
* The model chooses the plan; a weak model can still pick a wrong tool. Errors are fed back so it can retry.
* Isolation Forest is not included (it would need scikit-learn); IQR and z-score cover the common cases.
