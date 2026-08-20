const COLUMNS = [
  { key: "player_name",   label: "Player",       align: "left"  },
  { key: "team",          label: "Team",         align: "left"  },
  { key: "opponent",      label: "Opp",          align: "left"  },
  { key: "is_home",       label: "H/A",          align: "center"},
  { key: "line",          label: "Line",         align: "right" },
  { key: "ensemble_prob", label: "Model %",      align: "right" },
  { key: "over_implied",  label: "Implied %",    align: "right" },
  { key: "edge",          label: "Edge",         align: "right" },
  { key: "ev_per_unit",   label: "EV / $1",      align: "right" },
  { key: "kelly_qtr",     label: "Kelly (¼×)",   align: "right" },
  { key: "stake",         label: "Stake ($)",    align: "right" },  // computed
  { key: "bet_side",      label: "Side",         align: "center"},
  { key: "recommendation",label: "Signal",       align: "center"},
];

function sortRows(rows, key, dir, bankroll) {
  return [...rows].sort((a, b) => {
    const va = key === "stake" ? a.kelly_qtr * bankroll : a[key];
    const vb = key === "stake" ? b.kelly_qtr * bankroll : b[key];
    if (va === vb) return 0;
    const cmp = va < vb ? -1 : 1;
    return dir === "asc" ? cmp : -cmp;
  });
}

function Th({ col, sortKey, sortDir, onSort }) {
  const active = sortKey === col.key || (col.key === "stake" && sortKey === "kelly_qtr");
  const arrow  = active ? (sortDir === "asc" ? " ↑" : " ↓") : "";
  const align  = col.align === "right"  ? "text-right"
               : col.align === "center" ? "text-center"
               : "text-left";
  return (
    <th
      onClick={() => onSort(col.key === "stake" ? "kelly_qtr" : col.key)}
      className={`px-3 py-2 text-xs font-semibold uppercase tracking-wider
                  text-gray-400 cursor-pointer select-none whitespace-nowrap
                  hover:text-white transition-colors ${align}
                  ${active ? "text-blue-400" : ""}`}
    >
      {col.label}{arrow}
    </th>
  );
}

export default function PredictionsTable({ rows, bankroll, sortKey, sortDir, onSort }) {
  if (!rows || rows.length === 0) {
    return <p className="text-gray-500 text-sm mt-4">No props to display.</p>;
  }

  const sorted = sortRows(rows, sortKey, sortDir, bankroll);

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-800">
      <table className="w-full text-sm border-collapse">
        <thead className="bg-gray-900 sticky top-0 z-10">
          <tr>
            {COLUMNS.map(col => (
              <Th key={col.key} col={col}
                  sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
            ))}
          </tr>
        </thead>

        <tbody>
          {sorted.map((row, i) => {
            const isBet   = row.recommendation === "BET";
            const isOver  = row.bet_side === "OVER";
            const stake   = (row.kelly_qtr * bankroll).toFixed(2);
            const rowBg   = isBet
              ? "bg-emerald-950/40 hover:bg-emerald-900/30"
              : i % 2 === 0 ? "bg-gray-900/50 hover:bg-gray-800/40"
                            : "bg-gray-950   hover:bg-gray-800/40";

            return (
              <tr key={`${row.player_name}-${i}`} className={`transition-colors ${rowBg}`}>
                {/* Player */}
                <td className="px-3 py-2 font-medium text-white whitespace-nowrap">
                  {row.player_name}
                </td>
                {/* Team */}
                <td className="px-3 py-2 text-gray-400">{row.team}</td>
                {/* Opponent */}
                <td className="px-3 py-2 text-gray-400">{row.opponent}</td>
                {/* H/A */}
                <td className="px-3 py-2 text-center text-gray-400">
                  {row.is_home ? "H" : "A"}
                </td>
                {/* Line */}
                <td className="px-3 py-2 text-right tabular-nums">
                  {row.line}
                </td>
                {/* Model % */}
                <td className="px-3 py-2 text-right tabular-nums font-medium">
                  <span className={row.ensemble_prob >= 0.5 ? "text-emerald-400" : "text-gray-300"}>
                    {(row.ensemble_prob * 100).toFixed(1)}%
                  </span>
                </td>
                {/* Implied % */}
                <td className="px-3 py-2 text-right tabular-nums text-gray-400">
                  {(row.over_implied * 100).toFixed(1)}%
                </td>
                {/* Edge */}
                <td className="px-3 py-2 text-right tabular-nums font-semibold">
                  <span className={row.edge >= 0 ? "text-emerald-400" : "text-red-400"}>
                    {row.edge >= 0 ? "+" : ""}{(row.edge * 100).toFixed(2)}%
                  </span>
                </td>
                {/* EV */}
                <td className="px-3 py-2 text-right tabular-nums">
                  <span className={row.ev_per_unit >= 0 ? "text-emerald-300" : "text-red-400"}>
                    {row.ev_per_unit >= 0 ? "+" : ""}${row.ev_per_unit.toFixed(3)}
                  </span>
                </td>
                {/* Kelly ¼ */}
                <td className="px-3 py-2 text-right tabular-nums text-gray-300">
                  {(row.kelly_qtr * 100).toFixed(2)}%
                </td>
                {/* Stake — dynamic: bankroll × kelly_qtr */}
                <td className="px-3 py-2 text-right tabular-nums font-semibold">
                  <span className={isBet ? "text-emerald-300" : "text-gray-500"}>
                    ${stake}
                  </span>
                </td>
                {/* Side */}
                <td className="px-3 py-2 text-center">
                  <span className={`text-xs font-bold px-1.5 py-0.5 rounded
                    ${isOver
                      ? "bg-blue-900/60 text-blue-300"
                      : "bg-orange-900/60 text-orange-300"}`}>
                    {row.bet_side}
                  </span>
                </td>
                {/* Signal */}
                <td className="px-3 py-2 text-center">
                  {isBet ? (
                    <span className="inline-flex items-center gap-1 text-xs font-bold
                                     text-emerald-300 bg-emerald-900/50 px-2 py-0.5 rounded">
                      ✓ BET
                    </span>
                  ) : (
                    <span className="text-xs text-gray-600 font-medium">PASS</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="flex items-center justify-between px-4 py-2
                      bg-gray-900 border-t border-gray-800 text-xs text-gray-500">
        <span>
          {sorted.length} props shown
          {" · "}
          <span className="text-blue-400">
            {sorted.filter(r => r.recommendation === "BET").length} BET
          </span>
          {" · "}
          <span title="Raw sportsbook implied probability including vig (-110 = 52.38%)">
            Implied (w/vig): ~{sorted.length > 0
              ? (sorted[0].over_implied * 100).toFixed(1)
              : "52.4"}%
          </span>
        </span>
        <span>
          Total BET stake:{" "}
          <span className="text-emerald-400 font-semibold">
            ${sorted
              .filter(r => r.recommendation === "BET")
              .reduce((sum, r) => sum + r.kelly_qtr * bankroll, 0)
              .toFixed(2)}
          </span>
          {" "}of ${bankroll.toLocaleString()} bankroll
          <span className="text-gray-600 ml-1">(≤15% cap)</span>
        </span>
      </div>
    </div>
  );
}
