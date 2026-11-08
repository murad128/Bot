"""
Trade review & learning module.

What it does
* Per-signal post-mortem: classifies WHY a signal won or lost using what
  actually happened (MFE/MAE, milestones, exit type) and the conditions
  recorded at entry (component scores, tags, warnings).
* Aggregate review: finds tags / setups / sessions / score buckets /
  components whose results are materially worse than the baseline.

Anti-overfitting safeguards
* Minimum sample size per finding (default 30 resolved trades).
* Chronological split: a finding is only reported if the effect has the
  same sign in BOTH the earlier (in-sample) and the later (out-of-sample)
  part of the history.
* Wilson confidence interval on win rates (reported as evidence).
* Output is a list of RECOMMENDATIONS stored with status PENDING_REVIEW.
  Nothing is applied to live parameters. A human must approve, and any
  parameter change should be re-validated with `python main.py walkforward`.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.performance_analyzer import _get


def wilson(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def review_signal(rec: Any) -> Dict[str, Any]:
    outcome = _get(rec, "outcome")
    mfe = float(_get(rec, "mfe_r", 0) or 0)
    mae = float(_get(rec, "mae_r", 0) or 0)
    hits = int(_get(rec, "tp_hits", 0) or 0)
    status = _get(rec, "status")
    warnings = list(_get(rec, "warnings", []) or [])
    notes: List[str] = []
    if outcome == "CANCELLED":
        cause = (_get(rec, "review", {}) or {}).get("cancel_reason", "invalidated")
        notes.append(f"Cancelled before TP1: {cause}")
    elif status == "EXPIRED":
        notes.append("Time stop: market lacked follow-through within the holding window")
    elif outcome == "LOSS":
        if mfe < 0.3:
            notes.append("Immediate adverse move (MFE < 0.3R): entry timing or direction was wrong")
        elif mfe < 1.0:
            notes.append(f"Moved {mfe:.1f}R in favour then reversed before TP1")
        else:
            notes.append("Reached >1R but the stop was not protected")
    elif outcome == "WIN":
        if hits >= 3:
            notes.append("Full target reached")
        elif hits == 2:
            notes.append("TP2 reached, remainder stopped at trailed stop")
        elif hits == 1:
            notes.append("TP1 reached then stopped at breakeven: limited follow-through")
        else:
            notes.append("Closed in profit without reaching TP1")
    elif outcome == "BREAKEVEN":
        notes.append("Closed at breakeven")
    planned = float(_get(rec, "rr", 0) or 0)
    realised = float(_get(rec, "realized_r", 0) or 0)
    present = [w for w in warnings if "conflict" in w.lower() or "volatility" in w.lower() or "adx" in w.lower()]
    if outcome == "LOSS" and present:
        notes.append("Warnings present at entry: " + "; ".join(present[:3]))
    return {"notes": notes, "planned_rr": planned, "realised_r": realised, "mfe_r": mfe, "mae_r": mae}


@dataclass
class Finding:
    category: str
    text: str
    evidence: Dict[str, Any] = field(default_factory=dict)


def _resolved(records):
    return [r for r in records if _get(r, "outcome") in ("WIN", "LOSS", "BREAKEVEN")]


def _summ(rs):
    n = len(rs)
    w = sum(1 for r in rs if _get(r, "outcome") == "WIN")
    exp = sum(float(_get(r, "realized_r", 0) or 0) for r in rs) / n if n else 0.0
    return n, w, exp


class TradeReviewer:
    def __init__(self, min_samples: int = 30, min_effect_r: float = 0.15, split: float = 0.6):
        self.min_samples = min_samples
        self.min_effect_r = min_effect_r
        self.split = split

    def _groups(self, records) -> Dict[str, Dict[str, list]]:
        g: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            ctx = _get(r, "context", {}) or {}
            for t in ctx.get("tags", []):
                g["tag"][t].append(r)
            g["setup"][_get(r, "setup_type")].append(r)
            g["symbol"][_get(r, "symbol")].append(r)
            g["session"][_get(r, "session_bucket") or _get(r, "session")].append(r)
            sc = float(_get(r, "score", 0) or 0)
            bucket = "90-100" if sc >= 90 else "80-89" if sc >= 80 else "70-79" if sc >= 70 else "<70"
            g["score_bucket"][bucket].append(r)
            g["direction"][_get(r, "direction")].append(r)
        return g

    def analyze(self, records: List[Any]) -> List[Finding]:
        res = _resolved(records)
        res.sort(key=lambda r: _get(r, "closed_at") or _get(r, "created_at"))
        findings: List[Finding] = []
        if len(res) < self.min_samples:
            findings.append(Finding("info", f"Only {len(res)} resolved signals; at least {self.min_samples} are "
                                    "needed before the review produces recommendations.",
                                    {"resolved": len(res)}))
            return findings
        cut = int(len(res) * self.split)
        early, late = res[:cut], res[cut:]
        base_n, base_w, base_exp = _summ(res)
        _, _, base_e1 = _summ(early)
        _, _, base_e2 = _summ(late)
        g_all, g1, g2 = self._groups(res), self._groups(early), self._groups(late)

        for cat, groups in g_all.items():
            for key, rs in groups.items():
                n, w, exp = _summ(rs)
                if n < self.min_samples:
                    continue
                effect = exp - base_exp
                if abs(effect) < self.min_effect_r:
                    continue
                n1, _, e1 = _summ(g1[cat].get(key, []))
                n2, _, e2 = _summ(g2[cat].get(key, []))
                if n1 < 5 or n2 < 5:
                    continue
                consistent = (e1 - base_e1) * effect > 0 and (e2 - base_e2) * effect > 0
                if not consistent:
                    continue
                lo, hi = wilson(w, n)
                ev = {"n": n, "win_rate": round(w / n, 3), "win_rate_ci95": [round(lo, 3), round(hi, 3)],
                      "expectancy_r": round(exp, 3), "baseline_expectancy_r": round(base_exp, 3),
                      "in_sample_exp": round(e1, 3), "out_of_sample_exp": round(e2, 3), "n_is": n1, "n_oos": n2}
                if effect < 0:
                    txt = (f"{cat} '{key}' underperforms: expectancy {exp:+.2f}R vs baseline {base_exp:+.2f}R "
                           f"(n={n}, consistent in- and out-of-sample). Consider filtering or down-weighting it.")
                else:
                    txt = (f"{cat} '{key}' outperforms: expectancy {exp:+.2f}R vs baseline {base_exp:+.2f}R "
                           f"(n={n}, consistent in- and out-of-sample). Consider a higher weight for it.")
                findings.append(Finding(cat, txt, ev))

        # component discrimination: does the component score separate winners from losers?
        comp_names = set()
        for r in res:
            comp_names.update(((_get(r, "context", {}) or {}).get("components") or {}).keys())
        wins = [r for r in res if _get(r, "outcome") == "WIN"]
        losses = [r for r in res if _get(r, "outcome") == "LOSS"]
        if len(wins) >= 10 and len(losses) >= 10:
            for c in sorted(comp_names):
                mw = sum(((_get(r, "context") or {}).get("components") or {}).get(c, 0) for r in wins) / len(wins)
                ml = sum(((_get(r, "context") or {}).get("components") or {}).get(c, 0) for r in losses) / len(losses)
                if mw <= ml:
                    findings.append(Finding("component",
                                            f"Component '{c}' does not separate winners from losers "
                                            f"(mean score winners {mw:.2f} vs losers {ml:.2f}). Review its weight/logic.",
                                            {"mean_winners": round(mw, 3), "mean_losers": round(ml, 3),
                                             "n_wins": len(wins), "n_losses": len(losses)}))
        if not [f for f in findings if f.category != "info"]:
            findings.append(Finding("info", "No statistically consistent weaknesses found with the current sample.",
                                    {"resolved": base_n, "baseline_expectancy_r": round(base_exp, 3)}))
        return findings
