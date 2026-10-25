# The totals target: three classes, and a push is void

`OU-Cover` has three outcomes. The shipped `dataset_2012-26` table contains
**8,596 unders, 8,575 overs and 185 pushes** (17,356 rows; push is 1.066%).
The totals models keep all three classes. A push is **void**, never a side, and
the decision layer has to say so explicitly.

This note records why the target stays three-way, what the alternative would
cost, and the one measurement that would justify changing it.

## What the label is

`src/Process-Data/Create_Games.py` writes the label by exact comparison with the
closing line:

| `Points` vs `OU` | `OU-Cover` | meaning |
|---|---|---|
| `<` | 0 | under |
| `>` | 1 | over |
| `==` | 2 | push |

A push is an integer equality, not a near-miss. At a sportsbook the stake is
returned: the bet neither wins nor loses, and no price applies to it.

## What the model is asked to price

The bettable event is "over or under at this line, push void". Its full
distribution is the three probabilities. The two quantities a bettor needs are

- `P(push)` — the chance the bet is returned, and
- `P(over | no push)` and `P(under | no push)` — the conditional race that the
  odds are actually written against.

A three-class head gives all three directly. A two-class head fitted only on
non-push rows gives the conditional pair, and `P(push)` has to come from
somewhere else (the base rate, or a separate model).

## The two candidate targets

| | three classes (current) | two classes, push dropped |
|---|---|---|
| rows trained on | 17,356 | 17,171 (−1.07%) |
| target | under / over / push | under / over |
| outputs | `P(under), P(over), P(push)` | `P(over | no push)` |
| push at decision time | explicit, can be voided | must be re-injected from a prior |
| fitted push head | 185 examples — weakly identified | none |
| change required | none | rebuild the table, retrain both UO heads, re-derive the runner contract |

Dropping the pushes and treating them as void is coherent: for a bet at decimal
odds `b`, expected value per unit stake is

```
EV(over)  = P(over)  * (b - 1) - P(under)
EV(under) = P(under) * (b - 1) - P(over)
```

and the push contributes zero. If a two-class model is fitted on non-push rows,
multiplying its conditional EV by `P(no push)` recovers the same number. So the
choice is not about which arithmetic is correct — both can be made correct. It
is about what the model is asked to estimate and what can be checked.

## Decision

**Keep the three-way label.** Three reasons, in order of weight:

1. **The alternative cannot be measured.** `Data/OddsData.sqlite` stores the
   closing total *line* but not the closing over/under *price*. Without a price
   there is no ROI to score, so "three classes versus two" cannot be settled on
   the decision it feeds. The issue that proposed folding push said this
   explicitly. Changing a target on an unmeasurable preference is how a
   reasonable-sounding simplification becomes an unlogged regression.
2. **The push rows carry the outcome we must not invent.** Assigning 185 pushes
   to over or under fabricates a result; dropping them removes the only evidence
   that a void exists. The three-way head keeps the distinction and lets the
   decision layer choose what to do with it.
3. **The failure mode is in the consumer, not the target.** An argmax over three
   classes is not a bet. Before this change both runners treated "not under" as
   over, so a push-class prediction was printed and recorded as an **OVER** bet
   carrying `P(push)` as its confidence. That is the corruption the target
   discussion is really about, and it is fixed at the source: both runners now
   emit `PUSH` for class 2 and use class 2's own probability.

The cost of the three-way head is that `P(push)` is fitted on 185 rows and is
not a precise estimate. Treat it as a prior-informed quantity, not a sharp
probability: it is small, it moves the EV only through the void term, and it
must never be multiplied by odds.

## The decision rule the head feeds

For each game, with the three probabilities and the offered decimal price:

- `argmax` may be used for a **display lean**, and class 2 displays `PUSH`.
- A **bet decision** uses EV, not argmax. `PUSH` is never priced; it contributes
  zero to both EVs.
- A push-class argmax does **not** mean "bet the over". It means the single most
  likely outcome is a refund; the EV comparison still decides whether either
  side is worth a stake.

The Flask card styles `PUSH` as a neutral chip rather than reusing the over
colour, because a void is not a lean.

## What would reopen this

One measurement: closing over/under prices, both sides, joined to the same
`GameDate` and line. With those, score the three-class head and a
push-dropped binary head on identical test seasons by realised ROI, with a
paired bootstrap over games. If the binary target wins outside its standard
error, rebuild the table and switch both totals heads together — they share the
runner's column contract, so they cannot diverge.

Until then the target stays as it is, and the runners refuse to turn a push into
a side.
