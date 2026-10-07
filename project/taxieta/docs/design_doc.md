# TaxiETA design doc (Milestone 1) — reference solution

_Author: reference solution · Status: example. Replace the numbers with your own from `taxieta solve m1`._

## Problem
Riders see an ETA before and during a trip. When it's badly wrong, riders cancel, lose trust,
and complain. Dispatch also uses ETAs to chain the next pickup, and pricing uses them for
time-based fares. Today there's no estimate at all, or a crude distance ÷ constant speed.

## Why ML (the book's checklist)
| Criterion | ETA? |
| --- | --- |
| Learn from data | Yes: millions of historical trips with actual durations |
| Complex patterns | Yes: zone pair × time of week × live traffic interact nonlinearly; a lookup table only gets part of the way |
| Existing data | Yes: TLC trip records, plus our own completions |
| Predictive | Yes: duration of a future trip |
| Unseen data like training data | Mostly. Weekly seasonality repeats, but shifts happen (2020) → monitoring needed |
| Repetitive | Yes: every trip start |
| Cheap mistakes | Mostly: a 3-minute miss annoys, a 40-minute miss loses the rider. Tail errors matter more than the average |
| At scale | Yes: hundreds of thousands of predictions a day |
| Changing patterns | Yes: traffic, construction, seasons, pandemics → continual learning |

Non-ML baseline first (Ch. 6, phase 1): median duration per (pickup zone, dropoff zone,
hour-of-week), backing off to the zone pair, then the global median.

## Stakeholders and requirements
| Stakeholder | Requirement | Strictness |
| --- | --- | --- |
| Rider app PM | ETA shown in < 100 ms; doesn't jump around during the trip | Must-have (latency), nice-to-have (stability) |
| Dispatch / ops | Never *badly underestimate* long trips (driver chaining breaks) | Must-have |
| Pricing | Unbiased ETA on average (fare estimates) | Must-have |
| Platform / infra | Predictable resource use; no page for model issues outside SLOs | Must-have |
| Compliance | No sensitive attributes; comparable quality across boroughs | Must-have |

## Metrics
- **Business:** cancellation rate after ETA shown, support tickets about wrong ETAs, rider retention.
- **ML (primary):** % of trips with a prediction within ±20% of the actual duration. **Secondary:**
  MAE in minutes; p90 absolute error; worst-borough MAE.
- **Connection:** “% within ±20%” approximates “the rider felt the ETA was right”. We verify the
  link with an A/B test on cancellations before trusting it (Ch. 2).

## Task framing
- **Inputs known at pickup:** pickup timestamp, pickup zone, dropoff zone, history (computed
  before the cutoff), and live zone speeds as of the pickup time.
- **Never inputs:** dropoff time, metered distance, fare, tip (all only known after the trip).
- **Output:** duration in seconds, trained on log1p(duration), because durations are right-skewed.
- **Loss:** L1 on the log target (robust to jams and outliers). Alternatives: Huber, quantile.
- **Side task (LongTrip):** P(duration > 60 min), binary and heavily imbalanced (~1%).
- **Decoupled objectives:** ops' “don't underestimate long trips” is a separate model
  (LongTrip classifier, or a p90 quantile model) whose output is combined with the ETA at
  serving time. Each can be retuned and retrained on its own schedule without retraining the
  other (Ch. 2).

## Requirements
- **Reliability:** if the model is slow (> 100 ms) or broken, serve the heuristic and flag
  `fallback=true`. Stale streaming features count as missing, never silently reused.
- **Scalability:** peak ≈ `peak_rps_minute` from `taxieta solve m1`; design for 3× that. One model today, per-city models later.
- **Maintainability:** one feature module shared by training and serving; code, data window,
  and model versioned; the model store records lineage.
- **Adaptability:** drift monitoring + gated retraining (M7–M8); champion/challenger.

## SLOs
| SLO | Target |
| --- | --- |
| p50 latency | < 30 ms |
| p99 latency | < 150 ms |
| Availability | 99.9% (fallback counts as available) |
| Accuracy (24 h window) | % within ±20% no more than 10% below the offline value |

## Risks
Distribution shift (seasonality, events, 2020). Label delay (long trips arrive late). Leakage
from after-the-fact columns. Worse accuracy in outer boroughs (less data). Feedback loops if
ETAs influence which trips happen.

## Baseline plan
Heuristic on May 2019 (history Jan–Apr): see `reports/solutions/m1.md`. Every model must beat
it on MAE **and** on the worst slice to ship.
