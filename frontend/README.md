# Data Analysis Agent — Frontend

A plain HTML/CSS/JS chat interface for the backend agent (see the sibling
`backend/` project). No build step, no framework — just three files.

```
frontend/
├── index.html
├── style.css
└── script.js
```

## Setup

Open `script.js` and confirm `API_BASE` points at your running backend:

```js
const API_BASE = "http://localhost:5000";
```

That's the only configuration needed.

## Running it

1. Start the backend first (from the `backend/` folder):
   ```bash
   python run_api.py
   ```
2. Open `frontend/index.html` directly in a browser (double-click it), or
   serve it with any static file server, e.g.:
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   then visit `http://localhost:8080`.

The backend has CORS enabled for all origins, so it doesn't matter whether
the frontend is opened as a local file or served from a different port —
both work against the same running backend.

## What it does

- On load, fetches the bundled sample dataset's schema from
  `GET /api/datasets/sample/schema` and shows the columns in the sidebar.
- "Upload CSV" posts to `POST /api/upload`; the returned `dataset_id` is
  used for all subsequent questions in that session.
- Each chat message posts `{dataset_id, question}` to `POST /api/ask` and
  renders the answer, any generated chart inline, and an **Analysis** trace:
  every tool the agent ran, in order, with a check or cross, a one-line
  result, the arguments (filters shown as `region == West`), and a `parallel`
  tag for steps that ran concurrently.
- A status dot in the sidebar shows whether the backend is reachable.

## Notes

- This is intentionally a static, dependency-free frontend — easy to read
  and explain end-to-end for an interview.
- If `/api/ask` returns an error (e.g. missing `GROQ_API_KEY` on the
  backend), it's shown directly in the chat thread rather than failing
  silently.
