"""luck_test.py: does a daily trading rule beat a shuffled market?

    python3 luck_test.py prices.csv --rule rsi5 --perms 1000
    python3 luck_test.py prices.csv --rule sma --n 200 --from 2009-01-01

Needs only numpy. The CSV needs date, open, high, low, close columns (an adjclose column, if
present, scales OHLC to total return unless --price-only).

The test is neurotrader888's Monte Carlo permutation test (github.com/neurotrader888/mcpt, MIT):
the overnight gaps and the moves inside each bar are shuffled separately and the series is rebuilt.
That keeps the drift and the size of the moves but destroys their order. The same rule runs on
every shuffle and is scored the same way (profit factor of its daily log returns). p is the share
of shuffles, counting the real one, that score at least as well. A rule that only rides the drift
gets a p near 0.5; a timing edge pushes p toward 1 / (perms + 1).

Written by Proteus from the P-0082 probe (7 Oct 2026), where the numba version replicated five
YouTube strategy claims. Not advice; no costs or slippage are modelled.
"""
import argparse
import csv
import math

import numpy as np


def load(path, price_only=False, start=None):
    rows = list(csv.DictReader(open(path, newline="")))
    rows = [r for r in rows if r.get("close") not in (None, "", "null")]
    dates = [r["date"][:10] for r in rows]
    cols = {k: np.array([float(r[k]) for r in rows]) for k in ("open", "high", "low", "close")}
    if not price_only and rows and "adjclose" in rows[0]:
        k = np.array([float(r["adjclose"]) for r in rows]) / cols["close"]
        cols = {c: v * k for c, v in cols.items()}
    first = 0
    if start:
        first = next((i for i, d in enumerate(dates) if d >= start), len(dates))
    return dates, cols["open"], cols["high"], cols["low"], cols["close"], first


def sma(x, n):
    out = np.full(x.size, np.nan)
    if x.size >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rsi(c, n):
    """Wilder's RSI, as the backtest videos use it."""
    out = np.full(c.size, np.nan)
    if c.size <= n:
        return out
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    g, l = up[:n].mean(), dn[:n].mean()
    out[n] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    for i in range(n + 1, c.size):
        g = (g * (n - 1) + up[i - 1]) / n
        l = (l * (n - 1) + dn[i - 1]) / n
        out[i] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    return out


# Rules. Each takes (o, h, l, c, args) and returns daily log returns, index t = return over bar t.

def rule_sma(o, h, l, c, a):
    """Close above SMA(n): long from the next open. Close below: flat from the next open."""
    m = sma(c, a.n)
    sig = np.where(np.isnan(m), 0, np.where(c > m, 1, np.where(c < m, -1, 0)))
    pos = np.zeros(c.size)
    p = 0
    for t in range(c.size):
        pos[t] = p
        if sig[t] == 1:
            p = 1
        elif sig[t] == -1:
            p = 0
    rets = np.zeros(c.size)
    lo = np.log(o)
    # pos[t] was decided at the close of t-1 and filled at the open of t, so it earns open t to open t+1
    rets[:-1] = pos[:-1] * (lo[1:] - lo[:-1])
    return rets


def rule_rsi5(o, h, l, c, a):
    """Above SMA(200) and RSI(5) < 30: buy at the close. RSI(5) > 50: sell at the close."""
    r, m, lc = rsi(c, 5), sma(c, 200), np.log(c)
    rets = np.zeros(c.size)
    pos = 0
    for t in range(c.size - 1):
        if pos and r[t] > 50:
            pos = 0
        elif not pos and t >= 3 and not np.isnan(m[t]) and not np.isnan(r[t - 3]):
            if c[t] > m[t] and r[t] < 30:
                pos = 1
        rets[t + 1] = pos * (lc[t + 1] - lc[t])
    return rets


def rule_rsi2(o, h, l, c, a):
    """Above SMA(200) and RSI(2) < threshold: buy next open. Exit at the close above yesterday's high."""
    r, m = rsi(c, 2), sma(c, 200)
    rets = np.zeros(c.size)
    state = 0
    for t in range(1, c.size):
        if state == 1:
            rets[t] = math.log(c[t] / o[t])
            state = 2
        elif state == 2:
            rets[t] = math.log(c[t] / c[t - 1])
        if state == 2 and c[t] > h[t - 1]:
            state = 0
        if state == 0 and not np.isnan(m[t]) and not np.isnan(r[t]) and c[t] > m[t] and r[t] < a.threshold:
            state = 1
    return rets


RULES = {"sma": rule_sma, "rsi5": rule_rsi5, "rsi2": rule_rsi2}


def profit_factor(rets, start):
    r = rets[start:]
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    if loss == 0:
        return 1e9 if gain > 0 else 0.0
    return float(gain / loss)


def permute(o, h, l, c, start, rng):
    """Shuffle gaps and intrabar moves separately after bar `start`; bars up to it stay real."""
    lo, lh, ll, lc = np.log(o), np.log(h), np.log(l), np.log(c)
    i = np.arange(start + 1, c.size)
    gap = lo[i] - lc[i - 1]
    rh, rl, rc = lh[i] - lo[i], ll[i] - lo[i], lc[i] - lo[i]
    p1, p2 = rng.permutation(i.size), rng.permutation(i.size)
    gap, rh, rl, rc = gap[p2], rh[p1], rl[p1], rc[p1]
    # each new open = previous new close + gap; each new close = new open + intrabar close move
    steps = gap + rc
    pc = np.concatenate([lc[:start + 1], lc[start] + np.cumsum(steps)])
    po = np.concatenate([lo[:start + 1], pc[start:-1] + gap])
    ph = np.concatenate([lh[:start + 1], po[start + 1:] + rh])
    pl = np.concatenate([ll[:start + 1], po[start + 1:] + rl])
    return np.exp(po), np.exp(ph), np.exp(pl), np.exp(pc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--rule", choices=sorted(RULES), default="rsi5")
    ap.add_argument("--n", type=int, default=200, help="SMA length for --rule sma")
    ap.add_argument("--threshold", type=float, default=30.0, help="RSI(2) level for --rule rsi2")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--from", dest="start", help="score from this date; shuffles start here too")
    ap.add_argument("--warmup", type=int, default=300, help="bars ignored when scoring the full series")
    ap.add_argument("--price-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    dates, o, h, l, c, first = load(a.csv, a.price_only, a.start)
    rule = RULES[a.rule]
    shuffle_from = first if a.start else 0
    score_from = first if a.start else a.warmup
    real = profit_factor(rule(o, h, l, c, a), score_from)
    rng = np.random.default_rng(a.seed)
    vals = np.array([profit_factor(rule(*permute(o, h, l, c, shuffle_from, rng), a), score_from)
                     for _ in range(a.perms)])
    p = (1 + int((vals >= real).sum())) / (a.perms + 1)
    print(f"rule {a.rule}, {dates[score_from]} to {dates[-1]}, {c.size - score_from} bars scored")
    print(f"real profit factor {real:.3f}; shuffled median {np.median(vals):.3f}; "
          f"{int((vals >= real).sum())} of {a.perms} shuffles at least as good")
    print(f"p = {p:.4f}  ({'passes' if p < 0.05 else 'fails'} at 0.05)")


if __name__ == "__main__":
    main()
