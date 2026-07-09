#!/usr/bin/env python3
"""Multi-date SVI vs Heston vs flat-Black-Scholes benchmark (see
volsurface/benchmark.py for the methodology: out-of-sample RMSE on a
held-out 20% of strikes per maturity, per simulated date).

    python scripts/run_benchmark.py --n-dates 10
    python scripts/run_benchmark.py --n-dates 20 --n-starts 12 --max-iter 600 --n-points 1500  # slower, higher quality

This is the script that produces the headline "SVI RMSE / Heston RMSE /
Heston improvement vs flat vol" numbers referenced in the README. Runtime
scales roughly linearly in --n-dates and superlinearly in
--n-starts * --max-iter * --n-points (each is a full Heston re-calibration
per date) -- the defaults below are tuned to finish in a few minutes on a
laptop; the sandbox this repo was originally prototyped in had a much
harsher time budget per command, so the numbers quoted in the README come
from a reduced-resolution run (see README "Results" section for exact
settings) -- this script is what you re-run locally for the fully-powered
version.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volsurface import benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dates", type=int, default=10)
    parser.add_argument("--n-strikes", type=int, default=11)
    parser.add_argument("--n-starts", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--n-points", type=int, default=400)
    parser.add_argument("--out", default="benchmark_results.csv")
    args = parser.parse_args()

    heston_kwargs = dict(n_starts=args.n_starts, max_iter=args.max_iter, n_points=args.n_points)

    t0 = time.time()
    rows = []
    for i in range(args.n_dates):
        r = benchmark.run_single_date(i, n_strikes=args.n_strikes, heston_kwargs=heston_kwargs)
        rows.append(r)
        print(f"  date {i + 1}/{args.n_dates}  "
              f"SVI={r.svi_rmse_vol_pts * 100:6.3f}  Heston={r.heston_rmse_vol_pts * 100:6.3f}  "
              f"FlatBS={r.flat_bs_rmse_vol_pts * 100:6.3f} vol-pts   "
              f"({time.time() - t0:5.1f}s elapsed)")

    import pandas as pd
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(args.out, index=False)

    summary = benchmark.summarize(df)
    print("\n" + "=" * 60)
    print(f"Summary over {summary['n_dates']} simulated dates:")
    print(f"  SVI     mean out-of-sample RMSE: {summary['svi_rmse_mean'] * 100:.3f} vol-pts")
    print(f"  Heston  mean out-of-sample RMSE: {summary['heston_rmse_mean'] * 100:.3f} vol-pts")
    print(f"  Flat BS mean out-of-sample RMSE: {summary['flat_bs_rmse_mean'] * 100:.3f} vol-pts")
    print(f"  Heston improvement vs flat BS: {summary['heston_vs_flat_bs_improvement_pct']:.1f}%")
    print(f"  SVI improvement vs flat BS:    {summary['svi_vs_flat_bs_improvement_pct']:.1f}%")
    print(f"\nSaved per-date results: {args.out}")


if __name__ == "__main__":
    main()
