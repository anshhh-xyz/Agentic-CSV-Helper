# Data Analyst: Frontend

Plain HTML, CSS and JavaScript. No build step, no dependencies.

## Run

1. Start the backend from the `backend/` folder: `python run_api.py`
2. Open `index.html` in a browser (double-click it), or serve the folder:
   `python -m http.server 8080` then visit http://localhost:8080

If the backend is not at `http://localhost:5000`, change `API_BASE` at the top of `script.js`.

## What it does

- **Ask questions.** Answers appear as clean text (lists, tables and bold are rendered). Charts appear inline.
  Which tools the agent used is deliberately not shown.
- **Upload CSV.** If a file with the same name was analysed before but the contents differ, a dialog asks
  whether it is the *same dataset* (keeps remembered facts) or a *different dataset* (starts fresh).
- **Save changes to dataset.** When the agent cleaned the data (fill missing values, drop duplicates, and so on),
  a button appears under that answer. Without pressing it, the edit applies to that question only.
- **Memory.** Facts the agent remembers are listed in the sidebar; hover one and press x to forget it.
  The section only appears once something is remembered.
- The last dataset is reopened on refresh while the backend still has it; otherwise the bundled sample loads.

## Files

- `index.html`: structure
- `style.css`: theme (white, black, purple); colours are variables at the top
- `script.js`: behaviour

To remove the example questions on the empty screen, delete the `<div class="examples">` block in `index.html`.
