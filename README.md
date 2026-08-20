# NBA Prop Prediction & Quant Dashboard

> A full-stack quantitative trading tool that evaluates NBA player prop markets
> the same way a quant desk evaluates financial derivatives — by finding the delta
> between a calibrated probability model and the market's implied price.

---

## Screenshots

<!-- Insert dashboard screenshot here -->
<!-- ![Dashboard overview](docs/screenshots/dashboard.png) --><img width="1510" height="650" alt="Screenshot 2026-08-19 at 9 20 39 PM" src="https://github.com/user-attachments/assets/b9721084-877c-4f8a-87bd-3ff0310c80c3" />


<!-- Insert BET recommendations table here -->
<!-- ![BET filter active](docs/screenshots/bet_filter.png) -->

---

## Project Overview

This system treats a sportsbook prop market as a financial exchange. Each player
prop (e.g. *Luka Dončić over 28.5 points*) is a binary contract priced by the
book at an implied probability. The engine measures whether the book's price
deviates significantly from a calibrated ML ensemble's estimate — and when a
meaningful edge exists, it sizes a wager using the Kelly Criterion exactly as a
portfolio manager would size a position with positive expected return.

**The pipeline is fully automated:**

1. Live player prop lines and American odds are fetched from
   [The-Odds-API](https://the-odds-api.com) each morning.
2. An ensemble model (XGBoost + Scikit-Learn MLP) runs inference over 18
   rolling features per player to produce a calibrated P(over).
3. The optimizer computes vig-adjusted edge, EV per dollar, and a
   portfolio-capped quarter-Kelly stake for every prop.
4. A FastAPI backend serves the results as JSON; a React/Vite dashboard
   renders them in real time.

**Ensemble test-set performance (2024-25 season, chronological split):**

| Model | Accuracy | ROC-AUC | Log-Loss | Brier Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 81.43 % | 0.8637 | 0.4256 | 0.1356 |
| XGBoost | 88.06 % | 0.8620 | 0.3006 | 0.0893 |
| MLP Neural Network | 88.08 % | 0.8640 | 0.3010 | 0.0895 |
| **Ensemble (XGB 50% + MLP 50%)** | **88.22 %** | **0.8642** | **0.2992** | **0.0890** |

---

## Tech Stack

### Frontend
| Library | Version | Role |
|---|---|---|
| React | 19.2 | UI component model |
| Vite | 8.2 | Dev server & bundler |
| Tailwind CSS | 3.4 | Utility-first styling |
| Axios | 1.19 | HTTP client |

### Backend
| Library | Version | Role |
|---|---|---|
| FastAPI | 0.115 | REST API server |
| Uvicorn | 0.35 | ASGI runtime |
| Python | 3.13 | Runtime |

### Machine Learning
| Library | Version | Role |
|---|---|---|
| XGBoost | 3.4.1 | Gradient-boosted tree ensemble |
| Scikit-Learn | 1.9 | MLP, LR baseline, imputer, scaler |
| Pandas | 3.0.5 | Feature engineering & data wrangling |
| NumPy | 2.5.2 | Vectorised maths |
| PyArrow | 25.0 | Parquet I/O |

### Data Sources
| Source | Usage |
|---|---|
| [nba_api](https://github.com/swar/nba_api) | Historical game logs, team stats, schedules |
| [The-Odds-API](https://the-odds-api.com) | Live prop lines & American odds (DraftKings / FanDuel) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Ingestion                           │
│  nba_api → player/team game logs  │  The-Odds-API → live odds  │
│  data/raw/*.parquet               │  src/live_odds.py           │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
                ▼                             │
┌──────────────────────────┐                 │
│    Feature Engineering   │                 │
│    src/features.py       │                 │
│                          │                 │
│  18 rolling features:    │                 │
│  PTS/MIN/USG/TS% × 3/5/10│                 │
│  Days rest, home/away    │                 │
│  Opp DefRtg, team pace   │                 │
└──────────────┬───────────┘                 │
               │                             │
               ▼                             │
┌──────────────────────────┐                 │
│    Ensemble Inference    │                 │
│    src/predict_daily.py  │                 │
│                          │                 │
│  XGBoost (50%)           │                 │
│  + MLP 256→128→64 (50%)  │                 │
│  → ENSEMBLE_PROB         │                 │
└──────────────┬───────────┘                 │
               │                             │
               └──────────┬──────────────────┘
                          │
                          ▼
┌──────────────────────────────────────┐
│          Quant Optimizer             │
│          src/optimizer.py            │
│                                      │
│  Edge  = P_model − P_book(vigged)    │
│  EV    = P_model × decimal_odds − 1  │
│  Kelly = (b·p − q) / b  × 0.25      │
│  Cap   = scale stakes to ≤ 15% BR    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│          FastAPI Backend             │
│          src/api.py  :8000           │
│                                      │
│  GET /api/predictions/daily          │
│  POST /api/predictions/sgp           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│       React / Vite Dashboard         │
│       nba-dashboard/   :5173         │
│                                      │
│  Bankroll controls  │  BET filter    │
│  Min-line filter    │  Sort/rank     │
│  Live stake sizing  │  SGP builder   │
└──────────────────────────────────────┘
```

---

## Repository Structure

```
NBA Prediction Model/
├── data/
│   ├── raw/                 # Parquet game logs (nba_api)
│   ├── processed/           # Engineered feature matrices
│   └── predictions/         # Daily inference CSVs
├── models/                  # Saved model artefacts (.pkl, .json)
├── src/
│   ├── ingestion.py         # nba_api data fetching
│   ├── features.py          # Rolling feature engineering
│   ├── train.py             # LR / XGBoost / MLP training pipeline
│   ├── predict_daily.py     # Daily inference engine
│   ├── optimizer.py         # EV, Kelly Criterion, portfolio cap
│   ├── live_odds.py         # The-Odds-API ingestion
│   ├── sgp_engine.py        # Same Game Parlay correlation engine
│   └── api.py               # FastAPI REST server
├── nba-dashboard/           # React / Vite frontend
├── docs/
│   ├── SETUP.md             # Local dev setup guide
│   └── QUANT_MATH.md        # Mathematical methodology
├── main.py                  # End-to-end pipeline runner
├── requirements.txt
├── .env.example             # Environment variable template
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and create virtual environment
git clone <your-repo-url> && cd "NBA Prediction Model"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# edit .env and set: ODDS_API_KEY=your_key_here

# 3. Terminal 1 — FastAPI backend
set -a && source .env && set +a
venv/bin/uvicorn src.api:app --reload --port 8000

# 4. Terminal 2 — React frontend
cd nba-dashboard && npm install && npm run dev
```

Open **http://localhost:5173** — the dashboard loads live.

See [docs/SETUP.md](docs/SETUP.md) for a detailed walkthrough.

---

## Mathematical Methodology

See [docs/QUANT_MATH.md](docs/QUANT_MATH.md) for a rigorous treatment of:
- Probability calibration (Log-Loss, Brier Score)
- Vig-adjusted Expected Value
- Kelly Criterion for concurrent bet sizing
- Portfolio exposure cap implementation

---

## Validation

```bash
venv/bin/python validate_train.py   # model accuracy on held-out data
venv/bin/python validate_sgp.py     # SGP correlation engine smoke test
```
