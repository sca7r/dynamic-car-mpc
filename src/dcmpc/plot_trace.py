"""
Plot a CARLA trace CSV produced by carla_mpc.py.

Usage:
    python plot_trace.py                      # plots the newest carla_trace_*.csv
    python plot_trace.py carla_trace_X.csv    # plot a specific file

Shows, over time: cross-track error, heading error, speed, steering (commanded
vs applied), lateral-g, solve time, and brake events. The quickest way to see
*where* and *why* a run went wrong.
"""

from __future__ import annotations
import sys, glob, csv
import matplotlib
matplotlib.use("Agg") if "--live" not in sys.argv else None
import matplotlib.pyplot as plt


def load(path):
    rows = {}
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for c in cols:
            rows[c] = []
        for r in rd:
            for c in cols:
                v = r[c]
                try:
                    rows[c].append(float(v))
                except (ValueError, TypeError):
                    rows[c].append(float("nan"))
    return rows


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".csv")]
    if not files:
        cand = sorted(glob.glob("carla_trace_*.csv"))
        if not cand:
            print("No carla_trace_*.csv found. Run carla_mpc.py first.")
            sys.exit(1)
        files = [cand[-1]]
    path = files[0]
    print(f"Plotting {path}")
    d = load(path)
    t = d["t"]

    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t, d["cross_track_m"], "C0"); ax[0].set_ylabel("cross-track\n(m)")
    ax[0].axhline(0, color="0.7", lw=.8)
    ax[1].plot(t, d["heading_err_deg"], "C1"); ax[1].set_ylabel("heading err\n(deg)")
    ax[1].axhline(0, color="0.7", lw=.8)
    ax[2].plot(t, [v * 3.6 for v in d["vx"]], "C2"); ax[2].set_ylabel("speed\n(km/h)")
    ax[3].plot(t, d["mpc_delta_deg"], "C3", label="MPC delta (deg)")
    if any(s == s for s in d.get("steer", [])):  # not all NaN
        ax3b = ax[3].twinx()
        ax3b.plot(t, d["steer"], "C4", alpha=.6, label="CARLA steer [-1,1]")
        ax3b.set_ylabel("steer cmd")
    ax[3].set_ylabel("steering\n(deg)"); ax[3].set_xlabel("time (s)")

    # mark brake events
    braked = d.get("braked", [])
    for i, b in enumerate(braked):
        if b == 1:
            for a in ax:
                a.axvline(t[i], color="red", alpha=.25, lw=1)

    for a in ax:
        a.grid(alpha=.3)
    ax[0].set_title(f"CARLA MPC trace - {path}  (red lines = emergency brake)")
    fig.tight_layout()
    out = path.replace(".csv", ".png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")
    if "--live" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
