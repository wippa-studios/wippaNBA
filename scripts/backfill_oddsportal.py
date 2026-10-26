"""Backfill best-of-books moneyline odds from OddsPortal's season archives.

WHY THIS EXISTS
---------------
The profit question is settled at the best price a bettor can actually get, not
at one book's line. OddsPortal's results tables publish, for every NBA game
back to 1998/99, the **highest pre-game decimal odds across the bookmakers it
tracked** - one number per side (1X2 moneyline), each side possibly from a
different book. On the 14 Jun 2026 Finals game the table shows 1.54 / 2.75,
which are exactly the best closing prices across the eleven tracked books
(1xBet/GGBET/Megapari 1.54; Stake 2.75). That is a ~1.3% overround market
(~0.65% per side) versus the ~4-5% embedded in this repo's single-book tables.

SCOPE
-----
Seasons 2015/16 .. 2025/26 only - the span of `build_games()`. The main table is
moneyline only; spreads and totals live on each match's page and are NOT scraped
here.

DATE CONVENTION (the trap)
--------------------------
OddsPortal groups games by ITS OWN calendar day (a European day), and the
headers do NOT follow the browser timezone (verified: with
timezone_id=America/New_York the 14 Jun header still covers the game the US
logs date 13 Jun). For evening tip-offs the OddsPortal date is the US game date
PLUS ONE. The raw header date is stored verbatim as `Date_OP`; the mapping to
the US date happens in the join/validation step, never here.

OUTPUT
------
One CSV per season: Date_OP, Stage, Status, Home, Away, Home_Score, Away_Score,
O1, O2 (decimal, best-of-books), OddsPortal_URL. The caller validates against
`build_games()` before anything touches OddsData.sqlite.

USAGE
-----
    python scripts/backfill_oddsportal.py --seasons 2025-26 --max-pages 3   # smoke
    python scripts/backfill_oddsportal.py --seasons 2015-16..2025-26
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from playwright.sync_api import TimeoutError as PlaywrightTimeout  # noqa: E402

SEASON_SLUG = "nba-{y1}-{y2}"          # 2025-26 -> nba-2025-2026
BASE = "https://www.oddsportal.com/basketball/usa/{slug}/results/"
STATUS_WORDS = {"finished", "fin", "aot", "aet", "after ot", "afteret",
                "postponed", "canceled", "cancelled", "postp.", "awd",
                "walkover"}
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
INT_RE = re.compile(r"^\d+$")
DEC_RE = re.compile(r"^\d+\.\d{2}$")
DATE_HDR_RE = re.compile(
    r"^(Tomorrow|Today|Yesterday|\d{1,2} \w{3} \d{4})\s*(?:-\s*(.+))?$")

CHROMIUM_CANDIDATES = (
    "/home/Joel/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux*/headless_shell",
    "/home/Joel/.cache/ms-playwright/chromium-*/chrome-linux*/chrome",
)

# Tag the pagination Next (not the season-tab scroller's identical button):
# the pagination one has numeric page buttons among its siblings.
MARK_NEXT_JS = """() => {
  document.querySelectorAll('[data-op-next]').forEach(e => e.removeAttribute('data-op-next'));
  const nexts = [...document.querySelectorAll('main button')]
    .filter(b => /^next$/i.test((b.textContent || '').trim()) && !b.disabled);
  for (const n of nexts) {
    let c = n.parentElement;
    for (let i = 0; i < 5 && c; i++) {
      const nums = [...c.querySelectorAll('button')]
        .filter(x => /^\\d{1,3}$/.test((x.textContent || '').trim()));
      if (nums.length >= 2) { n.setAttribute('data-op-next', '1'); return true; }
      c = c.parentElement;
    }
  }
  return false;
}"""


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def pick_chromium() -> str:
    import glob
    for pattern in CHROMIUM_CANDIDATES:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    raise FileNotFoundError("no chromium under ~/.cache/ms-playwright; "
                            "run `playwright install chromium`")


def parse_link_text(text: str) -> dict | None:
    """'Finished 90 San Antonio Spurs - New York Knicks 94' -> parts."""
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    keep = [p for p in parts
            if p.lower() not in STATUS_WORDS and not TIME_RE.match(p) and p != "-"]
    ints = [p for p in keep if INT_RE.match(p)]
    teams = [p for p in keep if not INT_RE.match(p)]
    status = next((p for p in parts if p.lower() in STATUS_WORDS), "")
    if len(teams) != 2 or len(ints) < 2:
        return None
    return {"Status": status.title(), "Home_Score": int(ints[0]),
            "Away_Score": int(ints[1]), "Home": teams[0], "Away": teams[1]}


def season_pages(page, url: str) -> int:
    """Highest page number in the pagination strip (>= 1)."""
    nums = page.evaluate("""() => {
      const out = [];
      document.querySelectorAll('main button, main a').forEach(el => {
        const t = (el.textContent || '').trim();
        if (/^\\d{1,3}$/.test(t)) out.push(parseInt(t, 10));
      });
      return out;
    }""")
    return max([1] + [n for n in nums if 0 < n < 400])


def wait_dates_stable(page, max_wait_s: float = 12.0) -> None:
    """OddsPortal server-renders date headers in CET, then re-renders them in
    the browser timezone at hydration. Parsing mid-flip mixes the two calendars,
    which is exactly the silent-corruption failure this repo exists to prevent.
    Poll until the header set is identical across two samples."""
    def sample():
        return page.evaluate("""() => {
          const out = [];
          document.querySelectorAll('main *').forEach(el => {
            if (el.childElementCount !== 0) return;
            const t = (el.textContent||'').trim();
            if (/^(Tomorrow|Today|Yesterday|\\d{1,2} \\w{3} \\d{4})\\s*(?:-\\s*.+)?$/.test(t)
                && t.length < 45) out.push(t);
          });
          return out;
        }""")
    deadline = time.time() + max_wait_s
    prev = sample()
    while time.time() < deadline:
        time.sleep(1.5)
        cur = sample()
        if cur == prev:
            return
        prev = cur
    log("  WARNING: date headers never stabilized; dates may be mixed-calendar")


def parse_visible_rows(page) -> list[dict]:
    """Ordered walk: date headers are leaves matching the date pattern; each
    h2h link inherits the most recent preceding header. Post-hydration the
    headers render in the context timezone (set to America/New_York), so the
    date IS the US game date - validated against the ESPN close table."""
    return page.evaluate(r"""() => {
      const main = document.querySelector('main');
      const rows = [];
      let curDate = '';
      const seen = new Set();
      const rxDate = /^(Tomorrow|Today|Yesterday|\d{1,2} \w{3} \d{4})\s*(?:-\s*.+)?$/;
      main.querySelectorAll('*').forEach(el => {
        if (el.tagName === 'A' && (el.getAttribute('href')||'').includes('/h2h/')) {
          const href = el.getAttribute('href');
          if (seen.has(href)) return;
          seen.add(href);
          const row = el.parentElement;
          const odds = [];
          if (row) row.querySelectorAll('*').forEach(c => {
            const s = (c.textContent||'').trim();
            if (c.childElementCount === 0 && /^\d+\.\d\d$/.test(s)
                && !el.innerText.includes(s)) {
              const v = parseFloat(s);
              if (!odds.includes(v)) odds.push(v);
            }
          });
          rows.push({link_text: el.innerText || '', odds,
                     date_header: curDate, url: href});
          return;
        }
        if (el.childElementCount === 0) {
          const t = (el.textContent||'').trim();
          if (t && t.length < 45 && rxDate.test(t)) curDate = t;
        }
      });
      return rows;
    }""")


def crawl_season(page, season: str, max_pages: int | None, polite: float,
                 out_dir: Path) -> pd.DataFrame:
    y1 = int(season.split("-")[0])
    url = BASE.format(slug=SEASON_SLUG.format(y1=y1, y2=y1 + 1))
    page.goto(url, timeout=60_000, wait_until="domcontentloaded")
    page.wait_for_timeout(3_000)
    n_pages = season_pages(page, url)
    if max_pages:
        n_pages = min(n_pages, max_pages)
    log(f"{season}: {n_pages} pages")
    # A soft block or a boot race can land this URL on the CURRENT season's
    # page - the tell is upcoming games (kickoff times, no scores) and no
    # pagination. Detect it here, not at validation time.
    probe = parse_visible_rows(page)
    upcoming = sum(1 for r in probe if TIME_RE.match(r["link_text"].strip().split("\n")[0]))
    if n_pages <= 1 or (probe and upcoming > len(probe) / 2):
        raise RuntimeError(
            f"{season}: page 1 looks like the wrong page "
            f"({len(probe)} rows, {upcoming} upcoming, {n_pages} pages) - "
            "likely a soft block; retry with a fresh context")
    out: list[dict] = []
    rejects: list[str] = []
    seen_urls: set[str] = set()
    for pg_no in range(1, n_pages + 1):
        # Pagination mechanics (each proven against the live site):
        #   * a hash-only goto is a same-document navigation the SPA ignores;
        #   * a reload boots back to page 1, ignoring the hash;
        #   * the FIRST Next click after a boot renders the next page, and
        #     every later click updates the hash WITHOUT re-rendering.
        # The working cycle is: click Next (advances the hash), then reload -
        # the fresh boot renders the page the hash points at.
        if pg_no > 1:
            # The pagination strip can lag the reload; treat a missing Next as
            # retriable, and never end a season silently short.
            got_next = False
            for attempt in range(4):
                page.evaluate(MARK_NEXT_JS)
                if page.locator("[data-op-next]").count() > 0:
                    got_next = True
                    break
                page.wait_for_timeout(3_000)
            if not got_next:
                raise RuntimeError(
                    f"{season}: page {pg_no}: no enabled Next button after 4 "
                    "tries - pagination did not render; refusing to write a "
                    "truncated season")
            page.locator("[data-op-next]").click(timeout=5_000)
            page.wait_for_timeout(1_200)
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(6_000)
        wait_dates_stable(page)
        rows = parse_visible_rows(page)
        fresh = [r for r in rows if r["url"] not in seen_urls]
        if pg_no > 1 and not fresh:
            raise RuntimeError(
                f"{season}: page {pg_no} re-served only seen rows; the "
                "click-reload cycle failed, refusing to continue")
        for raw in rows:
            if raw["url"] in seen_urls:
                continue
            seen_urls.add(raw["url"])
            rec = parse_link_text(raw["link_text"])
            if rec is None:
                rejects.append(raw["link_text"].replace("\n", " | "))
                continue
            hdr = DATE_HDR_RE.match(raw["date_header"] or "")
            rec["Date_OP"] = hdr.group(1) if hdr else raw["date_header"]
            rec["Stage"] = (hdr.group(2) or "").strip() if hdr else ""
            rec["O1"], rec["O2"] = (raw["odds"] + [None, None])[:2]
            rec["OddsPortal_URL"] = raw["url"]
            rec["Season"] = season
            rec["Page"] = pg_no
            out.append(rec)
        log(f"  page {pg_no}: {len(fresh)} new games, "
            f"{len(rows) - len(fresh)} repeats skipped, "
            f"{len(rejects)} rejected total")
        time.sleep(polite + random.random() * polite)
    if rejects:
        (out_dir / f"rejects_{season}.txt").write_text(
            "\n".join(rejects) + "\n")
        log(f"  wrote {len(rejects)} rejected rows to rejects_{season}.txt")
    if not out:
        raise RuntimeError(
            f"{season}: 0 rows parsed from {url} - wrong URL, a bot block, or a "
            "page-structure change. Refusing to write an empty season.")
    if n_pages is not None and n_pages > 2 and len(out) < 0.5 * 51 * (n_pages - 1):
        raise RuntimeError(
            f"{season}: collected {len(out)} rows across {n_pages} pages - "
            "far short of the pagination promise; refusing to write a "
            "truncated season")
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="comma list (2015-16,2025-26) or range 2015-16..2025-26")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/opencode/oddsportal"))
    ap.add_argument("--polite", type=float, default=1.5,
                    help="base seconds between page loads")
    args = ap.parse_args()

    if ".." in args.seasons:
        lo, hi = args.seasons.split("..")
        def s2i(s): return int(s.split("-")[0])
        seasons = [f"{y}-{str(y + 1)[2:]}" for y in range(s2i(lo), s2i(hi) + 1)]
    else:
        seasons = [s.strip() for s in args.seasons.split(",")]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=pick_chromium(), headless=True)
        for season in seasons:
            csv_path = args.out_dir / f"oddsportal_{season}.csv"
            if csv_path.exists():
                log(f"{season}: already at {csv_path}, skipping")
                continue
            # A mid-season connection reset poisons the page: every later
            # goto fails with "interrupted by another navigation to
            # chrome-error://chromewebdata/" (observed 2017-18 onward). The
            # recovery is a FRESH context per attempt, with backoff between
            # attempts because the reset itself looks like rate limiting.
            df = None
            for attempt in range(1, 4):
                context = browser.new_context(timezone_id="America/New_York",
                                              locale="en-US")
                page = context.new_page()
                try:
                    df = crawl_season(page, season, args.max_pages,
                                      args.polite, args.out_dir)
                    break
                except Exception as exc:  # noqa: BLE001
                    log(f"{season}: attempt {attempt} FAILED "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
                    if attempt < 3:
                        wait_s = 60.0 * attempt
                        log(f"{season}: backing off {wait_s:.0f}s before "
                            "a fresh context")
                        time.sleep(wait_s)
                finally:
                    context.close()
            if df is None:
                log(f"{season}: giving up after 3 attempts; rerun the script "
                    "later - existing CSVs are never re-crawled")
                continue
            df.to_csv(csv_path, index=False)
            log(f"{season}: wrote {csv_path} ({len(df)} games, "
                f"{df.O1.notna().sum()} with odds)")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
