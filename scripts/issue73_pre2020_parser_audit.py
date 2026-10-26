"""Audit the pre-2020 injury-report layouts against the live CDN.

Issue #73 extended the parser to the 2018-19 / 2019-Q4 column grammars.  The
unit tests pin those grammars on generated PDFs; this script pins the REAL
files: it fetches a dated sample, parses each one, and records the accounting
(rows, unparsed lines, furniture, continuation lines) plus the statuses seen.
Nothing is written to Data/injury.sqlite -- this is an audit, not a capture.

Run it deliberately, with a cache directory, and commit the artifact:

    python scripts/issue73_pre2020_parser_audit.py

The script speaks only to the public CDN, waits between requests, and records
an HTTP failure as a failure (a 403 on 2018-12-14 is evidence about the archive
boundary, not a parse result).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "src" / "Process-Data" / "Get_Injury_Data.py"

# The sample spans every layout observed between the archive boundary and the
# first canonical month.  2018-12-14 is a deliberate negative control: the CDN
# has no report there, and the audit must record that, not invent a parse.
SAMPLES = (
    ("2018-12-14", "05PM"),
    ("2018-12-17", "05PM"),
    ("2019-01-15", "05PM"),
    ("2019-02-01", "05PM"),
    ("2019-03-01", "05PM"),
    ("2019-04-01", "05PM"),
    ("2019-10-25", "05PM"),
    ("2019-11-01", "05PM"),
    ("2019-11-15", "05PM"),
    ("2019-12-01", "05PM"),
    ("2019-12-20", "05PM"),
    ("2020-01-05", "05PM"),
)


def _load_parser():
    spec = importlib.util.spec_from_file_location("gid_audit", PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _url(day: str, slot: str) -> str:
    return ("https://ak-static.cms.nba.com/referee/injury/"
            f"Injury-Report_{day}_{slot}.pdf")


def _fetch(url: str, cache: Path, delay: float):
    name = url.rsplit("/", 1)[-1]
    path = cache / name
    if path.exists():
        return path.read_bytes(), 200, "cache"
    request = urllib.request.Request(
        url, headers={"User-Agent": "Legit-NBA-Predictions parser audit"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        return b"", exc.code, "http_error"
    finally:
        time.sleep(delay)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data, status, "fetched"


def _parser_sha256() -> str:
    return hashlib.sha256(PARSER.read_bytes()).hexdigest()


def audit(cache: Path, delay: float) -> dict:
    parser = _load_parser()
    records = []
    for day, slot in SAMPLES:
        url = _url(day, slot)
        data, status, source = _fetch(url, cache, delay)
        record = {"date": day, "slot": slot, "url": url,
                  "http_status": status, "source": source}
        if status != 200:
            record["error"] = f"HTTP {status}"
            records.append(record)
            continue
        if not data.startswith(b"%PDF"):
            record["error"] = "response is not a PDF"
            records.append(record)
            continue
        record["sha256"] = hashlib.sha256(data).hexdigest()
        try:
            parsed = parser.parse_injury_pdf(data)
        except Exception as exc:  # noqa: BLE001 - the audit records failures
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        statuses = Counter(row["status"] for row in parsed["rows"])
        record.update({
            "rows": len(parsed["rows"]),
            "unparsed": len(parsed["unparsed"]),
            "furniture": len(parsed["furniture"]),
            "continuation_lines": parsed["continuation_lines"],
            "header_lines": parsed["header_lines"],
            "error": parsed["error"],
            "statuses": dict(sorted(statuses.items())),
            "unparsed_examples": [
                {"line_no": row["line_no"], "reason": row["reason"],
                 "raw_text": row["raw_text"][:160]}
                for row in parsed["unparsed"][:5]
            ],
        })
        records.append(record)

    parseable = [r for r in records if "rows" in r]
    return {
        "issue": 73,
        "title": "pre-2020 injury-report parser audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction": ("python scripts/issue73_pre2020_parser_audit.py "
                         "--cache-dir /tmp/issue73_pdfs"),
        "parser_path": str(PARSER.relative_to(ROOT)),
        "parser_sha256": _parser_sha256(),
        "samples": len(records),
        "parseable": len(parseable),
        "total_rows": sum(r["rows"] for r in parseable),
        "total_unparsed": sum(r["unparsed"] for r in parseable),
        "total_furniture": sum(r["furniture"] for r in parseable),
        "records": records,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Issue #73 — pre-2020 parser audit",
        "",
        "Real CDN PDFs parsed with `src/Process-Data/Get_Injury_Data.py`. "
        "Counts only; the raw files are not committed.",
        "",
        f"Generated {payload['generated_utc']}. Parser sha256 "
        f"`{payload['parser_sha256'][:16]}…`.",
        "",
        "| date | HTTP | rows | unparsed | furniture | continuation | statuses |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in payload["records"]:
        if "rows" not in record:
            lines.append(
                f"| {record['date']} | {record['http_status']} | — | — | — | — | "
                f"{record.get('error', '')} |")
            continue
        statuses = ", ".join(
            f"{name} {count}" for name, count in record["statuses"].items())
        lines.append(
            f"| {record['date']} | {record['http_status']} | "
            f"{record['rows']} | {record['unparsed']} | "
            f"{record['furniture']} | {record['continuation_lines']} | "
            f"{statuses} |")
    lines += [
        "",
        f"**Totals:** {payload['parseable']} parseable samples, "
        f"{payload['total_rows']:,} rows, "
        f"{payload['total_unparsed']} unparsed lines, "
        f"{payload['total_furniture']} furniture lines.",
        "",
        "The 2018-12-14 row is a negative control: the CDN returns 403, which "
        "is the left edge of the archive, not a parser failure.  A zero "
        "`unparsed` column on every parseable report is the no-silent-drop "
        "claim for this sample.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="/tmp/issue73_pdfs",
                        help="where fetched PDFs are cached")
    parser.add_argument("--out-prefix", default="results/issue73_pre2020_parser_audit",
                        help="JSON and Markdown artifact prefix")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="seconds between CDN requests")
    args = parser.parse_args(argv)

    payload = audit(Path(args.cache_dir), args.delay)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(
        render_markdown(payload), encoding="utf-8")
    print(f"{payload['parseable']}/{payload['samples']} parseable; "
          f"{payload['total_rows']:,} rows; "
          f"{payload['total_unparsed']} unparsed")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
