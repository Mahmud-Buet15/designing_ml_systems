# TaxiETA design doc (Milestone 1)

_Author: · Date: · Status: draft_

## Problem
What goes wrong for riders and the business today without accurate ETAs?

## Why ML
Go through the book's checklist: learn, complex patterns, existing data, predictive, unseen data like the training data, repetitive, cheap mistakes, scale, changing patterns.

## Stakeholders and requirements
| Stakeholder | Requirement | Must-have / nice-to-have |
| --- | --- | --- |
| Rider app PM | | |
| Dispatch / ops | | |
| Pricing | | |
| Platform / infra | | |
| Compliance | | |

## Metrics
- Business:
- ML: MAE (min), % within ±20%
- How they connect (and how you'll verify it, e.g. an A/B test):

## Task framing
- Inputs known at pickup:
- Output:
- Loss / objective:
- Side task (LongTrip > 60 min):
- Decoupled objectives? (one model with a combined loss, or two models whose scores are combined)

## Requirements
- Reliability:
- Scalability (peak req/s from the data):
- Maintainability:
- Adaptability:

## SLOs
| SLO | Target |
| --- | --- |
| p50 latency | |
| p99 latency | |
| Availability | |
| Accuracy (24 h window) | |

## Risks

## Baseline plan
Heuristic: median duration per (pickup zone, dropoff zone, hour-of-week). MAE on May 2019: ___
