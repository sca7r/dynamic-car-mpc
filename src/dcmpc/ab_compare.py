"""
ab_compare.py - head-to-head comparison of two CARLA traces for the
return-to-lane overshoot.

Usage:
    python ab_compare.py baseline.csv treatment.csv

Reads the columns: t, cross_track_m, heading_err_deg, mpc_delta_deg, vx,
obstacle (all present in a standard dcmpc CARLA trace). Prints the four
headline overshoot metrics side by side, plus a per-obstacle-pass breakdown,
plus a one-line verdict telling you what the result means for the NEXT knob.

The metrics, and why each one matters for THIS overshoot:
  worst |cross-track|   the outcome you are trying to shrink (run summary's number)
  peak  |heading error| the mechanism - the overshoot is a HEADING overshoot
  max   |steering|      is the MPC still SATURATING (pegging MAX_STEER ~28.6 deg)?
  steering saturation % what fraction of ticks sit at the steering limit
  max speed             control variable - should be ~unchanged (we only touched
                        the planner ramp), confirming the A/B isolated one thing
"""

import csv
import sys

SAT_DEG = 28.5          # |mpc_delta_deg| at/above this == steering saturated
POST_PASS_S = 4.0       # seconds after an obstacle clears to look for the overshoot


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            def g(k):
                v = r.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            rows.append(dict(t=g("t"), cross=g("cross_track_m"),
                             herr=g("heading_err_deg"), delta=g("mpc_delta_deg"),
                             vx=g("vx"), obs=g("obstacle")))
    return [r for r in rows if r["t"] is not None]


def overall(rows):
    def peak(key):
        best, bt = 0.0, None
        for r in rows:
            v = r[key]
            if v is not None and abs(v) > best:
                best, bt = abs(v), r["t"]
        return best, bt
    wc, wct = peak("cross")
    ph, pht = peak("herr")
    md, _ = peak("delta")
    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    sat = 100.0 * sum(1 for d in deltas if abs(d) >= SAT_DEG) / max(1, len(deltas))
    vmax = max((r["vx"] for r in rows if r["vx"] is not None), default=0.0) * 3.6
    return dict(worst_cross=wc, worst_cross_t=wct, peak_herr=ph, peak_herr_t=pht,
                max_delta=md, sat_pct=sat, vmax_kmh=vmax)


def passes(rows):
    """Find obstacle passes (contiguous obstacle==1) and, for each, the peak
    |cross| and |heading| in the pass + POST_PASS_S window (the overshoot)."""
    out, i, n = [], 0, len(rows)
    while i < n:
        if rows[i]["obs"] == 1:
            j = i
            while j < n and rows[j]["obs"] == 1:
                j += 1
            t0, t1 = rows[i]["t"], rows[j - 1]["t"]
            win = [r for r in rows if t0 <= r["t"] <= t1 + POST_PASS_S]
            pc = max((abs(r["cross"]) for r in win if r["cross"] is not None), default=0.0)
            ph = max((abs(r["herr"]) for r in win if r["herr"] is not None), default=0.0)
            out.append(dict(t0=t0, t1=t1, peak_cross=pc, peak_herr=ph))
            i = j
        else:
            i += 1
    return out


def fmt_row(label, a, b, unit="", lower_better=True):
    d = b - a
    arrow = ""
    if abs(d) > 1e-9:
        better = (d < 0) if lower_better else (d > 0)
        arrow = "  ✓ better" if better else "  ✗ worse"
    return f"{label:<28}{a:>10.2f}{unit:<5}{b:>10.2f}{unit:<5}{d:>+9.2f}{arrow}"


def main():
    if len(sys.argv) != 3:
        print("usage: python ab_compare.py baseline.csv treatment.csv")
        sys.exit(1)
    ba, tr = load(sys.argv[1]), load(sys.argv[2])
    A, B = overall(ba), overall(tr)

    print(f"\n{'':28}{'BASELINE':>13}{'TREATMENT':>15}{'Δ':>10}")
    print("-" * 70)
    print(fmt_row("worst |cross-track|", A["worst_cross"], B["worst_cross"], " m"))
    print(fmt_row("peak |heading error|", A["peak_herr"], B["peak_herr"], " °"))
    print(fmt_row("max |steering|", A["max_delta"], B["max_delta"], " °"))
    print(fmt_row("steering saturation", A["sat_pct"], B["sat_pct"], " %"))
    print(fmt_row("max speed", A["vmax_kmh"], B["vmax_kmh"], " kmh",
                  lower_better=False))  # informational; want it ~unchanged

    print("\nper obstacle pass (peak cross / peak heading in pass+4s window):")
    pa, pb = passes(ba), passes(tr)
    for k in range(max(len(pa), len(pb))):
        a = pa[k] if k < len(pa) else None
        b = pb[k] if k < len(pb) else None
        ax = f"{a['peak_cross']:.2f} m / {a['peak_herr']:.0f}°" if a else "   -"
        bx = f"{b['peak_cross']:.2f} m / {b['peak_herr']:.0f}°" if b else "   -"
        print(f"  pass {k+1:<2}  baseline {ax:>16}    treatment {bx:>16}")

    # verdict: what does this mean for the NEXT knob?
    print("\nverdict:")
    cross_better = B["worst_cross"] < A["worst_cross"] - 0.10
    still_sat = B["sat_pct"] > 1.0 and B["max_delta"] >= SAT_DEG
    vchg = abs(B["vmax_kmh"] - A["vmax_kmh"])
    if vchg > 3.0:
        print(f"  ! speed changed by {vchg:.0f} km/h - the A/B is NOT clean; the runs")
        print(f"    differed in something other than the ramp. Re-run more carefully.")
    if cross_better and not still_sat:
        print("  ✓ Overshoot shrank AND steering no longer saturates.")
        print("    The controller tracks the return cleanly. Ship these settings.")
    elif cross_better and still_sat:
        print("  ~ Overshoot shrank but steering still saturates. The change helped;")
        print("    the controller is still pegging MAX_STEER on the return. Target the")
        print("    saturation directly next: raise the steering-rate penalty")
        print("    INPUT_RATE_COST[1] and/or the sideslip damping STATE_COST[4] (vy),")
        print("    and/or widen CROSS_BAND so it stops fighting the off-centre line.")
        print("    (Yaw-rate weight STATE_COST[5] is the more indirect lever.)")
    else:
        print("  ✗ The change did NOT shrink the overshoot and steering still")
        print("    saturates. This is a controller-tuning problem, not a reference")
        print("    one. Damp the return directly: raise INPUT_RATE_COST[1]")
        print("    (steering-rate) and STATE_COST[4] (vy), and confirm saturation")
        print("    duty drops WITHOUT worst cross-track creeping back up.")
    print()


if __name__ == "__main__":
    main()
