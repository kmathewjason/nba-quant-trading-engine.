# Quantitative Methodology

**NBA Prop Prediction & Quant Dashboard**

> This document details the mathematical and statistical foundations of the
> system for a reader with a quantitative finance or applied mathematics
> background. The problem of pricing sports prop bets is structurally identical
> to pricing binary options: we are estimating P(event) and comparing it to
> the market's implied P(event) embedded in the quoted odds.

---

## 1. Probability Modelling

### 1.1 Problem Framing

The prediction target is a binary random variable:

$$Y_i = \mathbf{1}\left[\text{player } i \text{ scores} > L_i\right]$$

where $L_i$ is the sportsbook's prop line for player $i$. We estimate
$\hat{p}_i = P(Y_i = 1 \mid \mathbf{x}_i)$ using an ensemble of probabilistic
classifiers trained on historical player game logs with a strict chronological
train/test split to prevent data leakage.

### 1.2 Feature Vector

Each inference row $\mathbf{x}_i \in \mathbb{R}^{18}$ encodes:

| Feature group | Features | Windows |
|---|---|---|
| Scoring | `PTS_ROLL` | 3, 5, 10 games |
| Minutes | `MIN_ROLL` | 3, 5, 10 games |
| Usage rate | `USG_ROLL` | 3, 5, 10 games |
| True shooting | `TS_ROLL` | 3, 5, 10 games |
| Context | `DAYS_REST`, `IS_HOME` | — |
| Matchup | `OPP_DEFRTG_ROLL`, `TEAM_PACE_ROLL` | 5, 10 games |

All rolling windows use `shift(1)` before aggregating — the current game's
outcome is never included in any feature used to predict that game.

Usage rate approximation:
$$\text{USG} \approx \frac{\text{FGA} + 0.44 \cdot \text{FTA} + \text{TOV}}{\text{MIN}} \times 48$$

True shooting percentage:
$$\text{TS\%} = \frac{\text{PTS}}{2(\text{FGA} + 0.44 \cdot \text{FTA})}$$

### 1.3 Ensemble Architecture

The ensemble is a soft-vote combination of two independently trained models:

$$\hat{p}_{\text{ens}} = 0.5 \cdot \hat{p}_{\text{XGB}} + 0.5 \cdot \hat{p}_{\text{MLP}}$$

**XGBoost** — gradient-boosted trees tuned via `TimeSeriesSplit` cross-validation:

| Hyperparameter | Tuned value |
|---|---|
| `max_depth` | 3 |
| `learning_rate` | 0.03 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |

**MLP Neural Network** (Scikit-Learn `MLPClassifier`) — three fully-connected
layers with ReLU activations, Adam optimiser, and early stopping:

$$\mathbb{R}^{18} \xrightarrow{\text{ReLU}} \mathbb{R}^{256} \xrightarrow{\text{ReLU}} \mathbb{R}^{128} \xrightarrow{\text{ReLU}} \mathbb{R}^{64} \xrightarrow{\sigma} [0, 1]$$

### 1.4 Calibration Objectives

Models are selected and compared on metrics that reward **probability
calibration**, not just binary accuracy:

**Log-Loss** (cross-entropy):

$$\mathcal{L}_{\text{log}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

A model that outputs $\hat{p}_i = 0.9$ for every positive example and is
correct 90% of the time achieves perfect accuracy but poor log-loss if those
positives are only 60% likely — log-loss penalises overconfidence.

**Brier Score** (mean squared probability error):

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

The Brier Score is the $L^2$ distance between forecast probabilities and
outcomes. A score of 0 is perfect; a naive coin-flip predictor scores 0.25.

**Test-set results (2024-25 season, chronological split at 80th percentile):**

| Model | Accuracy | ROC-AUC | Log-Loss | Brier Score |
|---|---|---|---|---|
| Logistic Regression | 81.43 % | 0.8637 | 0.4256 | 0.1356 |
| XGBoost | 88.06 % | 0.8620 | 0.3006 | 0.0893 |
| MLP (256→128→64) | 88.08 % | 0.8640 | 0.3010 | 0.0895 |
| **Ensemble** | **88.22 %** | **0.8642** | **0.2992** | **0.0890** |

The ensemble improves over each individual model on all calibration metrics.
A Brier Score of 0.089 means average squared error of 8.9 percentage points —
meaningfully better than the 0.135 of the logistic baseline.

---

## 2. Vig-Adjusted Expected Value

### 2.1 Converting American Odds to Decimal

American moneyline odds are converted to the decimal (European) format used
throughout the EV and Kelly calculations:

$$d = \begin{cases} \dfrac{A}{100} + 1 & A > 0 \\ \dfrac{100}{|A|} + 1 & A < 0 \end{cases}$$

Examples: $-110 \Rightarrow d = 1.\overline{909}$, $+150 \Rightarrow d = 2.50$

### 2.2 Implied Probability and the Vigorish

The sportsbook's **raw implied probability** is:

$$p_{\text{implied}} = \frac{1}{d}$$

At $-110$ on both sides: $p_{\text{implied}} = \frac{1}{1.\overline{909}} = 0.5238$

The book's total overround (vig) for a two-sided market is:

$$\text{overround} = p_{\text{over}} + p_{\text{under}} = 0.5238 + 0.5238 = 1.0476$$

The excess above 1.0 (4.76%) is the house edge. Standard $-110$/$-110$ pricing
implies a **4.55% hold** per betting event for the sportsbook.

> **Implementation note:** The system uses the **raw vigged probability**
> ($p_{\text{implied}} = 1/d = 52.38\%$) as the `OVER_IMPLIED` baseline for
> edge calculation. This is the correct hurdle — the model must beat 52.38%,
> not the no-vig 50.00%, to generate a bet recommendation.

### 2.3 Expected Value

For a bet on the over at decimal odds $d$ where the model assigns probability $\hat{p}$:

$$\text{EV} = \hat{p} \cdot (d - 1) - (1 - \hat{p}) = \hat{p} \cdot d - 1$$

Decomposed: $\hat{p} \cdot (d-1)$ is the expected profit when correct;
$(1-\hat{p}) \cdot 1$ is the expected loss per dollar when incorrect.

| $\hat{p}$ | $d$ at $-110$ | EV per $1 | Interpretation |
|---|---|---|---|
| 0.50 | 1.909 | $-$0.045 | Paying full vig, no edge |
| 0.5238 | 1.909 | $\approx 0.000$ | Breakeven — vig exactly absorbed |
| 0.60 | 1.909 | $+$0.145 | +14.5 cents per dollar — strong edge |
| 0.70 | 1.909 | $+$0.336 | +33.6 cents per dollar — very strong |

The system requires **both** `EDGE ≥ 4%` (model vs. vigged implied) **and**
`EV ≥ $0.05 per $1` before issuing a `BET` recommendation. Requiring both
filters eliminates micro-edges that are real but too small to overcome
execution friction.

### 2.4 Edge Definition

$$\text{Edge} = \hat{p}_{\text{model}} - p_{\text{implied}}$$

where $p_{\text{implied}} = 1/d$ is the **raw book probability including vig**.
This is the true alpha — the excess probability the model sees above what the
market prices.

---

## 3. Kelly Criterion for Portfolio Sizing

### 3.1 Single-Bet Kelly

The Kelly Criterion maximises the expected geometric growth rate of a bankroll
by solving:

$$f^* = \frac{bp - q}{b}$$

where:
- $f^*$ — fraction of bankroll to wager
- $b = d - 1$ — net odds (profit per $1 risked)
- $p = \hat{p}_{\text{model}}$ — model's estimated win probability
- $q = 1 - p$ — estimated loss probability

The formula is derived by maximising $E[\log(W)]$ where $W$ is the resulting
wealth. This is equivalent to maximising the long-run compound growth rate,
which Kelly (1956) proved is achieved by this exact fractional allocation.

**Worked example** — Luka Dončić over 28.5 at $-115$:
- $d = 100/115 + 1 = 1.870$, $b = 0.870$
- $\hat{p} = 0.72$, $q = 0.28$
- $f^* = (0.870 \times 0.72 - 0.28) / 0.870 = 0.397$ ← full Kelly

Full Kelly is highly aggressive and maximises long-run growth but incurs severe
variance. The system applies a **quarter-Kelly** multiplier:

$$f_{\text{qtr}} = 0.25 \times f^*$$

Quarter-Kelly reduces drawdowns to roughly $\frac{1}{16}$ of full-Kelly
drawdowns while retaining approximately 75% of the optimal compound growth rate.
It is the standard conservative scaling used in quantitative sports finance.

### 3.2 The Concurrent Events Problem

The Kelly formula assumes **sequential, independent bets**. On a typical NBA
slate there are 8–15 games with 100+ props firing simultaneously. Naively summing
independent Kelly fractions across all bets produces absurd total allocations
(e.g. $7,400 on a $1,000 bankroll).

This is the **simultaneous events problem** — Kelly fractions from independent
analyses cannot be additively composed for concurrent positions without
violating the compound-growth-maximisation proof's assumptions.

### 3.3 Portfolio Exposure Cap

The system resolves this with a proportional normalisation algorithm:

```
1. Compute f*_qtr for every prop independently via the Kelly formula.
2. Sum all f*_qtr values for props flagged BET:
       F_total = Σ f*_qtr(i)  for all i where RECOMMENDATION = BET
3. If F_total > cap (15%):
       scale = cap / F_total             # scale ∈ (0, 1)
       f*_qtr(i) ← f*_qtr(i) × scale   # proportional reduction
4. Stake_i = f*_qtr(i) × Bankroll
```

This preserves **relative sizing** — a prop with twice the edge still receives
twice the stake — while enforcing the hard constraint:

$$\sum_{i \in \text{BET}} \text{Stake}_i \leq 0.15 \times \text{Bankroll}$$

The 15% cap is a risk management parameter with a clear interpretation: it is
the maximum daily drawdown in the worst case where every single flagged bet
loses simultaneously. At 15% daily exposure, a 10-day losing streak (extremely
unlikely with a model at 88%+ accuracy) would reduce a $1,000 bankroll to
$\approx $197 — painful but not ruinous. The cap can be adjusted via the
`max_daily_exposure` parameter in `optimizer.evaluate_props()`.

### 3.4 Ruin Probability

Under the quarter-Kelly system with 15% daily exposure cap, the probability of
bankroll ruin (drawdown to a fraction $\epsilon$ of initial wealth) over an
infinite time horizon is bounded by:

$$P(\text{ruin}) \leq \left(\frac{q}{p}\right)^{B/s}$$

where $B$ is the bankroll and $s$ is the unit stake. With $p > 0.52$ (the
breakeven threshold enforced by the EV filter) and quarter-Kelly sizing, the
long-run expected growth per bet is positive and ruin is avoided in expectation.

---

## 4. Same Game Parlay Correlation Adjustment

Standard parlay probability assumes leg independence:
$$P(\text{parlay}) = \prod_{i=1}^{n} p_i$$

Within a single game, player outcomes are **not** independent — a high-pace
game simultaneously lifts all players' point totals, inducing positive
correlation between legs from the same team.

The system models this using a **Gaussian copula** on the historical joint
distribution of player scores:

1. Compute the Pearson correlation matrix $\Sigma$ over rolling player scoring
   vectors using `pandas.DataFrame.corr()`.
2. For a two-leg SGP, adjust the joint probability using the bivariate normal CDF:

$$P(X_1 > L_1, X_2 > L_2) = \Phi_2\!\left(\Phi^{-1}(p_1),\, \Phi^{-1}(p_2);\, \rho_{12}\right)$$

where $\Phi_2$ is the bivariate standard normal CDF and $\rho_{12}$ is the
empirical correlation between the two players' scoring series.

3. EV and Kelly are then computed on the adjusted joint probability against the
   book's SGP price.

---

## 5. Calibration and Leakage Prevention

**Chronological split:** The train/test boundary is set at the 80th percentile
of the sorted game date index. No future data ever influences features or
targets for past games.

**Rolling window shift:** All rolling statistics use `.shift(1)` before
`.rolling(w).mean()`, ensuring each row's features reflect only data available
*before* that game was played.

**Pipeline:** The `sklearn.Pipeline` with `SimpleImputer → StandardScaler`
is fit exclusively on training data. The same fitted transformer is serialised
to `models/prop_imputer.pkl` and `models/prop_scaler.pkl` and applied
(not re-fitted) at inference time.
