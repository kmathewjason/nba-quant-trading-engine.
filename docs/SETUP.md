# Local Development Setup

**NBA Prop Prediction & Quant Dashboard**

> Estimated time: ~10 minutes on a clean macOS machine.
> Tested on macOS 15 (Apple Silicon / ARM).

---

## Prerequisites

| Tool | Minimum version | Check |
|---|---|---|
| Python | 3.11+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | any | `git --version` |

---

## Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd "NBA Prediction Model"
```

---

## Step 2 — Python virtual environment

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it — you must do this in every new terminal session
source venv/bin/activate
```

Your prompt should now show `(venv)` at the start. All subsequent Python
commands assume the venv is active.

---

## Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs XGBoost, Scikit-Learn, FastAPI, Uvicorn, Pandas, nba_api, and
all other dependencies pinned to their tested versions.

> **Apple Silicon note:** XGBoost and Scikit-Learn install native ARM wheels
> automatically. PyTorch is listed in requirements but the ML pipeline uses
> `sklearn.MLPClassifier` — no CUDA or Metal setup is required.

---

## Step 4 — Configure the API key

The only secret the project needs is a [The-Odds-API](https://the-odds-api.com)
key for fetching live prop lines. The free developer tier provides ~500 requests/
month, which is sufficient for daily NBA prop pulls.

```bash
# Copy the template
cp .env.example .env

# Open it in your editor and replace the placeholder
nano .env          # or: code .env / vim .env / open -a TextEdit .env
```

Your `.env` file should look like this (replace with your actual key):

```
ODDS_API_KEY=your_key_here
```

> **Security:** `.env` is listed in `.gitignore` and will never be committed.
> Never paste your key directly into any source file.

---

## Step 5 — Verify models are trained

The repository ships with pre-trained model artefacts in `models/`. Verify
they are present:

```bash
ls models/
# Expected: prop_lr.pkl  prop_xgb.pkl  prop_mlp.pkl
#           prop_imputer.pkl  prop_scaler.pkl  prop_meta.json
```

If the `models/` directory is empty (e.g. after a fresh clone without LFS),
run the full training pipeline:

```bash
# Takes 5–10 minutes on first run (NBA API rate limits apply)
venv/bin/python main.py --season 2024-25
```

---

## Step 6 — Start the FastAPI backend

Open a dedicated terminal for the API server. The `set -a && source .env`
step exports your `.env` variables into the shell before Uvicorn starts.

```bash
# Terminal 1 — run from the project root
set -a && source .env && set +a
venv/bin/uvicorn src.api:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

Test it:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Interactive API docs are available at **http://localhost:8000/docs**.

---

## Step 7 — Install frontend dependencies and start Vite

Open a second terminal for the React dev server.

```bash
# Terminal 2
cd nba-dashboard
npm install          # first time only — installs node_modules
npm run dev
```

You should see:

```
  VITE v8.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## Step 8 — Open the dashboard

Navigate to **http://localhost:5173** in your browser.

The dashboard will automatically call `http://localhost:8000/api/predictions/daily`
on load. If the API server is running and a game date has predictions, the table
populates immediately.

---

## Daily workflow

On any given game day, both servers must be running. The dashboard automatically
uses today's date if no date is specified in the Game Date field.

```bash
# Refresh today's predictions (re-runs inference for today's schedule)
venv/bin/python -m src.predict_daily
```

The API caches results per `(game_date, prop_line, over_odds, under_odds,
min_line)` tuple. Click **↻ Refresh** in the dashboard or restart the API
server to invalidate the cache.

---

## Validation scripts

```bash
# Confirm model artefacts load and produce expected accuracy
venv/bin/python validate_train.py

# Smoke-test the SGP correlation engine
venv/bin/python validate_sgp.py
```

---

## Troubleshooting

### `Address already in use` on port 8000

Another Uvicorn process is running. Kill it:

```bash
kill -9 $(lsof -ti :8000)
```

Then restart with the `source .env` command in Step 6.

### Dashboard shows red error banner

The React app cannot reach the API. Confirm:
1. The FastAPI server is running in Terminal 1.
2. You see `Uvicorn running on http://127.0.0.1:8000` in that terminal.
3. `curl http://localhost:8000/health` returns `{"status":"ok"}`.

### `No games found` in API logs

The NBA season may be in off-season, or the date has no scheduled games.
Try specifying a historical date with confirmed games:

```bash
# In the dashboard Game Date field:
2025-04-13

# Or via the API directly:
curl "http://localhost:8000/api/predictions/daily?game_date=2025-04-13"
```

### Live odds not loading (fallback to model lines)

Check that your `.env` file has the correct key and was sourced before Uvicorn:

```bash
echo $ODDS_API_KEY    # should print your key, not empty
```

If empty, re-run `set -a && source .env && set +a` in the API terminal and
restart Uvicorn.

### Apple Silicon / macOS segfault during training

The training pipeline uses `multiprocessing.set_start_method("spawn")` to
prevent Metal/OpenMP conflicts between XGBoost and Scikit-Learn on ARM. If you
hit a segfault, ensure you are running `main.py` (not importing `src.train`
directly) so the spawn guard is active.

---

## Full retrain (new season)

```bash
venv/bin/python main.py \
  --season 2024-25 \
  --bankroll 1000 \
  --prop-line 19.5 \
  --min-edge 0.04
```
