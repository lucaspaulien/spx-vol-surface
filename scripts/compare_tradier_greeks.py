#!/usr/bin/env python3
"""Compare this repo's closed-form Black-Scholes Greeks against Tradier's
broker-reported Greeks for the same live option chain.

Requires a Tradier API token (sandbox or production) in the environment:

    pip install -e ".[compare]"
    export TRADIER_TOKEN="your_token_here"
    python scripts/compare_tradier_greeks.py --ticker SPY --expiry-index 2

By default this hits the sandbox market-data host. Pass --production if
your token is a live brokerage token instead of a sandbox token.

Methodology: for each strike, we take Tradier's own reported mid IV and feed
it into OUR closed-form Black-Scholes greeks() (src/volsurface/black_scholes.py).
This isolates the Greeks formula itself (same vol in, do the numbers match)
from any difference in implied-vol *estimation* between Tradier's engine and
ours -- the fairest apples-to-apples check. A second, looser check
(--reprice-iv) additionally re-inverts IV from Tradier's own mid price using
this repo's Newton-Raphson solver, to see how much of any gap is vol
estimation vs. the Greeks formula itself.

This is the one script in the repo that talks to a paid-data-adjacent third
party API; everything else (svi.py, heston.py, greeks.py) only ever
consumes yfinance or the synthetic generator.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volsurface import black_scholes as bs  # noqa: E402

SANDBOX_HOST = "https://sandbox.tradier.com"
PRODUCTION_HOST = "https://api.tradier.com"

GREEKS = ["delta", "gamma", "vega", "theta"]


def tradier_get(requests, host, token, path, params):
    resp = requests.get(
        f"{host}{path}",
        params=params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_rows(requests, host, token, ticker, expiry_index, option_type, rate, div_yield, reprice_iv):
    quote = tradier_get(requests, host, token, "/v1/markets/quotes", {"symbols": ticker})
    last = quote["quotes"]["quote"].get("last")
    if not last:
        raise RuntimeError(f"Tradier returned no usable last price for {ticker!r} "
                            "(market may be closed, or the symbol never traded today).")
    spot = float(last)

    expirations = tradier_get(requests, host, token, "/v1/markets/options/expirations",
                               {"symbol": ticker, "includeAllRoots": "true"})
    dates = expirations["expirations"]["date"]
    if isinstance(dates, str):
        dates = [dates]
    if not dates:
        raise RuntimeError(f"Tradier returned no expirations for {ticker!r}.")
    if expiry_index >= len(dates):
        raise RuntimeError(f"--expiry-index {expiry_index} out of range, only {len(dates)} expiries available.")
    expiry = dates[expiry_index]

    ttm = (date.fromisoformat(expiry) - date.today()).days / 365.0
    if ttm <= 0:
        raise RuntimeError(f"chosen expiry {expiry} is today or in the past, pick a different --expiry-index.")

    chain = tradier_get(requests, host, token, "/v1/markets/options/chains",
                         {"symbol": ticker, "expiration": expiry, "greeks": "true"})
    options = chain["options"]["option"] if chain["options"] else []
    if isinstance(options, dict):  # Tradier collapses a single-contract result to a bare object
        options = [options]

    types = ["call", "put"] if option_type == "both" else [option_type]
    rows = []
    for opt in options:
        if opt["option_type"] not in types:
            continue
        g = opt.get("greeks") or {}
        iv = g.get("mid_iv")
        if not iv or any(g.get(k) is None for k in GREEKS):
            continue

        strike = float(opt["strike"])
        bid, ask = opt.get("bid"), opt.get("ask")

        vol_for_our_calc = iv
        if reprice_iv and bid and ask and bid > 0 and ask > 0:
            mid_price = 0.5 * (bid + ask)
            try:
                vol_for_our_calc = float(bs.implied_vol(
                    mid_price, spot, strike, rate, div_yield, ttm, option_type=opt["option_type"]))
            except Exception:
                pass  # fall back to Tradier's own IV if our solver fails to converge on this quote

        our = bs.greeks(spot, strike, rate, div_yield, vol_for_our_calc, ttm, option_type=opt["option_type"])
        # bs.greeks() returns theta per YEAR and vega per 1.0 (100 vol points)
        # of calendar time / vol -- the natural units for the closed-form
        # calculus. Brokers (Tradier included) report theta per CALENDAR DAY
        # and vega per 1 VOL POINT (0.01). Comparing the raw values would
        # make every quote look like a ~365x theta / ~100x vega "bug" that is
        # actually just a unit convention mismatch, not a Greeks error.
        our_theta_per_day = float(our["theta"]) / 365.0
        our_vega_per_point = float(our["vega"]) / 100.0
        rows.append({
            "option_type": opt["option_type"],
            "strike": strike,
            "tradier_iv": iv,
            "our_iv": vol_for_our_calc,
            **{f"t_{k}": g[k] for k in GREEKS},
            "o_delta": float(our["delta"]), "o_gamma": float(our["gamma"]),
            "o_vega": our_vega_per_point, "o_theta": our_theta_per_day,
        })
    return spot, expiry, ttm, rows


def print_table(rows):
    header = (f"{'type':>4} {'strike':>8} {'IV(T)':>7} {'IV(us)':>7} | "
              f"{'delta(T)':>9} {'delta(us)':>9} {'diff':>7} | "
              f"{'gamma(T)':>9} {'gamma(us)':>9} {'diff':>8} | "
              f"{'vega/pt(T)':>10} {'vega/pt(us)':>11} {'diff':>7} | "
              f"{'theta/day(T)':>12} {'theta/day(us)':>13} {'diff':>8}")
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['option_type']:>4} {r['strike']:8.1f} {r['tradier_iv']:7.4f} {r['our_iv']:7.4f} | "
              f"{r['t_delta']:9.4f} {r['o_delta']:9.4f} {r['o_delta'] - r['t_delta']:7.4f} | "
              f"{r['t_gamma']:9.5f} {r['o_gamma']:9.5f} {r['o_gamma'] - r['t_gamma']:8.5f} | "
              f"{r['t_vega']:10.3f} {r['o_vega']:11.3f} {r['o_vega'] - r['t_vega']:7.3f} | "
              f"{r['t_theta']:12.4f} {r['o_theta']:13.4f} {r['o_theta'] - r['t_theta']:8.4f}")


def print_summary(rows):
    print(f"\n{len(rows)} strikes compared.")
    print(f"{'greek':>7} {'mean|diff|':>11} {'max|diff|':>11} {'mean|diff|/mean|T|':>20}")
    for k in GREEKS:
        diffs = [abs(r[f"o_{k}"] - r[f"t_{k}"]) for r in rows]
        t_mags = [abs(r[f"t_{k}"]) for r in rows]
        mean_diff = statistics.mean(diffs)
        max_diff = max(diffs)
        mean_t = statistics.mean(t_mags) or 1e-12
        print(f"{k:>7} {mean_diff:11.5f} {max_diff:11.5f} {mean_diff / mean_t:19.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--rate", type=float, default=0.045, help="risk-free rate (continuous, annualised)")
    parser.add_argument("--div-yield", type=float, default=0.013, help="dividend yield (continuous, annualised)")
    parser.add_argument("--expiry-index", type=int, default=2,
                         help="which expiry in the chain to use, 0=nearest (default: 2, skips weeklies)")
    parser.add_argument("--option-type", choices=["call", "put", "both"], default="both")
    parser.add_argument("--strike-range-pct", type=float, default=0.10,
                         help="only compare strikes within +/- this fraction of spot (default 0.10)")
    parser.add_argument("--reprice-iv", action="store_true",
                         help="also re-invert IV from Tradier's mid price with our own Newton-Raphson "
                              "solver, instead of trusting Tradier's mid_iv as the vol input")
    parser.add_argument("--production", action="store_true", help="use api.tradier.com instead of sandbox")
    parser.add_argument("--save-csv", default=None, help="optional path to save the comparison table as CSV")
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("Error: the 'requests' package is required for this script.\n"
              "Install it with: pip install -e \".[compare]\"", file=sys.stderr)
        sys.exit(1)

    import os
    token = os.environ.get("TRADIER_TOKEN")
    if not token:
        print("Error: set TRADIER_TOKEN in your environment first "
              "(free sandbox token: https://developer.tradier.com/).", file=sys.stderr)
        sys.exit(1)

    host = PRODUCTION_HOST if args.production else SANDBOX_HOST

    try:
        spot, expiry, ttm, rows = fetch_rows(requests, host, token, args.ticker, args.expiry_index,
                                              args.option_type, args.rate, args.div_yield, args.reprice_iv)
    except requests.exceptions.HTTPError as e:
        print(f"\nError: Tradier API request failed: {e}", file=sys.stderr)
        print("Common causes: invalid/expired token, or a sandbox token used with --production "
              "(or vice versa).", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\nError: could not reach Tradier ({type(e).__name__}: {e}). "
              "Check your network connection and try again.", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"{args.ticker} spot (Tradier): {spot}")
    print(f"Using expiry: {expiry} (index {args.expiry_index} of the available chain), ttm={ttm:.4f}y")
    print(f"Rate/div-yield used for OUR Greeks: --rate {args.rate} --div-yield {args.div_yield} "
          "(these are our own assumptions, not Tradier's -- a mismatch here shows up as a delta/rho "
          "diff that looks like a formula bug but is actually a parameter mismatch).")

    rows = [r for r in rows if abs(r["strike"] - spot) / spot < args.strike_range_pct]
    rows.sort(key=lambda r: (r["option_type"], r["strike"]))

    if not rows:
        print(f"\nNo strikes with usable Tradier greeks within +/-{args.strike_range_pct:.0%} of spot came back "
              "(chain may be illiquid, greeks=true unsupported on this plan, or the strike range is too tight). "
              "Try --production, a different --expiry-index, or a wider --strike-range-pct.", file=sys.stderr)
        sys.exit(1)

    print_table(rows)
    print_summary(rows)
    vol_source = "our own Newton-Raphson inversion of Tradier mid price" if args.reprice_iv else "Tradier's own mid_iv"
    print(f"\nVol input = {vol_source} for both Greek columns -- diffs isolate the Greeks formula "
          "(dividend-yield convention, day-count, discounting), not vol estimation.")

    if args.save_csv:
        import csv
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved comparison table: {out_path}")


if __name__ == "__main__":
    main()
