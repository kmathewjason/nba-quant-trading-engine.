import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import PredictionsTable from "./components/PredictionsTable";

const API_BASE = "http://localhost:8000";

// Min-line options: maps label → API value sent to backend
const MIN_LINE_OPTIONS = [
  { label: "All players",    value: 0   },
  { label: "Line ≥ 5.5 pts", value: 5.5 },
  { label: "Line ≥ 10.5 pts",value: 10.5},
  { label: "Line ≥ 15.5 pts",value: 15.5},
  { label: "Line ≥ 20.5 pts",value: 20.5},
];

export default function App() {
  const [bankroll, setBankroll]     = useState(1000);
  const [propLine, setPropLine]     = useState(19.5);
  const [overOdds, setOverOdds]     = useState(-110);
  const [underOdds, setUnderOdds]   = useState(-110);
  const [gameDate, setGameDate]     = useState("");
  const [minLine, setMinLine]       = useState(5.5);
  const [data, setData]             = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [filter, setFilter]         = useState("ALL");   // ALL | BET
  const [sortKey, setSortKey]       = useState("edge");
  const [sortDir, setSortDir]       = useState("desc");

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        prop_line:  propLine,
        over_odds:  overOdds,
        under_odds: underOdds,
        min_line:   minLine,
      });
      if (gameDate) params.append("game_date", gameDate);

      const res = await axios.get(`${API_BASE}/api/predictions/daily?${params}`);
      setData(res.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Could not connect to the API. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  }, [propLine, overOdds, underOdds, gameDate, minLine]);

  // Load on mount
  useEffect(() => { fetchPredictions(); }, [fetchPredictions]);

  const rows = data?.predictions ?? [];
  const visible = filter === "BET"
    ? rows.filter(r => r.recommendation === "BET")
    : rows;

  return (
    <div className="min-h-screen p-6 max-w-screen-2xl mx-auto">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            NBA Prop Prediction Dashboard
          </h1>
          {data && (
            <p className="text-sm text-gray-400 mt-0.5">
              {data.game_date} · {data.total_props} props ·{" "}
              <span className="text-emerald-400 font-medium">
                {data.bet_count} BET
              </span>
              {" "}recommendations
            </p>
          )}
        </div>
        <button
          onClick={fetchPredictions}
          disabled={loading}
          className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50
                     text-sm font-medium transition-colors"
        >
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      {/* ── Controls ────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-5">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 uppercase tracking-wide">Bankroll ($)</span>
          <input
            type="number" min="1" value={bankroll}
            onChange={e => setBankroll(Number(e.target.value))}
            className="input"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 uppercase tracking-wide">Fallback Line</span>
          <input
            type="number" step="0.5" value={propLine}
            onChange={e => setPropLine(Number(e.target.value))}
            className="input"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 uppercase tracking-wide">Over Odds</span>
          <input
            type="number" value={overOdds}
            onChange={e => setOverOdds(Number(e.target.value))}
            className="input"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 uppercase tracking-wide">Under Odds</span>
          <input
            type="number" value={underOdds}
            onChange={e => setUnderOdds(Number(e.target.value))}
            className="input"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 uppercase tracking-wide">Game Date</span>
          <input
            type="date" value={gameDate}
            onChange={e => setGameDate(e.target.value)}
            className="input"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 uppercase tracking-wide">Min Line</span>
          <select
            value={minLine}
            onChange={e => { setMinLine(Number(e.target.value)); }}
            className="input"
          >
            {MIN_LINE_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 uppercase tracking-wide">Show</span>
          <select
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="input"
          >
            <option value="ALL">All props</option>
            <option value="BET">BET only</option>
          </select>
        </label>
      </div>

      {/* ── Error ───────────────────────────────────────────────────────── */}
      {error && (
        <div className="mb-4 p-3 rounded bg-red-900/50 border border-red-700 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* ── Table ───────────────────────────────────────────────────────── */}
      {loading && !data ? (
        <p className="text-gray-500 text-sm">Fetching predictions…</p>
      ) : (
        <PredictionsTable
          rows={visible}
          bankroll={bankroll}
          sortKey={sortKey}
          sortDir={sortDir}
          onSort={(key) => {
            if (key === sortKey) setSortDir(d => d === "asc" ? "desc" : "asc");
            else { setSortKey(key); setSortDir("desc"); }
          }}
        />
      )}
    </div>
  );
}
