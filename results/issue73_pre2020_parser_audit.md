# Issue #73 — pre-2020 parser audit

Real CDN PDFs parsed with `src/Process-Data/Get_Injury_Data.py`. Counts only; the raw files are not committed.

Generated 2026-10-26T05:25:20.676744+00:00. Parser sha256 `875a15e9b7611459…`.

| date | HTTP | rows | unparsed | furniture | continuation | statuses |
|---|---|---|---|---|---|---|
| 2018-12-14 | 403 | — | — | — | — | HTTP 403 |
| 2018-12-17 | 200 | 89 | 0 | 10 | 4 | Available 1, Doubtful 1, Out 74, Probable 3, Questionable 10 |
| 2019-01-15 | 200 | 91 | 0 | 14 | 7 | Available 5, Out 65, Probable 7, Questionable 14 |
| 2019-02-01 | 200 | 121 | 0 | 18 | 3 | Available 2, Doubtful 2, Out 92, Probable 9, Questionable 16 |
| 2019-03-01 | 200 | 85 | 0 | 16 | 4 | Available 5, Out 69, Probable 2, Questionable 9 |
| 2019-04-01 | 200 | 78 | 0 | 14 | 5 | Available 1, Doubtful 6, Out 61, Probable 1, Questionable 9 |
| 2019-10-25 | 200 | 63 | 0 | 17 | 12 | Doubtful 1, Out 47, Probable 3, Questionable 12 |
| 2019-11-01 | 200 | 94 | 0 | 17 | 10 | Available 5, Doubtful 1, Out 81, Probable 3, Questionable 4 |
| 2019-11-15 | 200 | 132 | 0 | 19 | 60 | Available 6, Doubtful 3, Out 102, Probable 12, Questionable 9 |
| 2019-12-01 | 200 | 132 | 0 | 19 | 55 | Available 15, Doubtful 3, Out 107, Probable 4, Questionable 3 |
| 2019-12-20 | 200 | 138 | 0 | 24 | 2 | Available 6, Doubtful 1, Out 113, Probable 7, Questionable 11 |
| 2020-01-05 | 200 | 117 | 0 | 14 | 2 | Available 2, Out 91, Probable 9, Questionable 15 |

**Totals:** 11 parseable samples, 1,140 rows, 0 unparsed lines, 182 furniture lines.

The 2018-12-14 row is a negative control: the CDN returns 403, which is the left edge of the archive, not a parser failure.  A zero `unparsed` column on every parseable report is the no-silent-drop claim for this sample.
