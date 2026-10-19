# Validation alert-policy selection

Threshold chosen only on the 20 held-out full run-to-failure validation engines. This is not test-set lifecycle alert evidence.

Selected threshold: **20 cycles**; actionable horizon 30; persistence 3 consecutive cycles; minimum history 20 cycles.

Early false alarms: 0/20 engines (0.0%). Timely detections: 20/20 (100.0%). Missed failures: 0/20. Mean timely lead: 11.75 cycles.

An early false alarm is an engine's first persistent alert when its reconstructed RUL is over 30 cycles. A timely detection is first alert at RUL 1–30. Failure-cycle-only alerts count as misses. Denominator for all event rates is the number of full validation engines.

| Threshold | Early false alarms | Timely detections | Missed | Mean timely lead (cycles) |
|---:|---:|---:|---:|---:|
| 5 | 0/20 | 11/20 | 9 | 3.73 |
| 10 | 0/20 | 17/20 | 3 | 6.12 |
| 15 | 0/20 | 19/20 | 1 | 10.05 |
| 20 | 0/20 | 20/20 | 0 | 11.75 |
| 25 | 1/20 | 19/20 | 0 | 15.58 |
| 30 | 2/20 | 18/20 | 0 | 17.06 |
| 35 | 4/20 | 16/20 | 0 | 19.50 |
| 40 | 7/20 | 13/20 | 0 | 19.54 |
| 45 | 8/20 | 12/20 | 0 | 21.33 |
| 50 | 9/20 | 11/20 | 0 | 23.00 |
| 55 | 11/20 | 9/20 | 0 | 23.67 |
| 60 | 15/20 | 5/20 | 0 | 23.40 |
| 65 | 16/20 | 4/20 | 0 | 24.50 |
| 70 | 17/20 | 3/20 | 0 | 27.33 |
| 75 | 20/20 | 0/20 | 0 | — |
| 80 | 20/20 | 0/20 | 0 | — |
| 85 | 20/20 | 0/20 | 0 | — |
| 90 | 20/20 | 0/20 | 0 | — |
| 95 | 20/20 | 0/20 | 0 | — |
| 100 | 20/20 | 0/20 | 0 | — |
