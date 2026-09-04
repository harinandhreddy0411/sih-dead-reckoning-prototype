# GNSS-Denied Dead Reckoning — Pitch Notes

## (a) Problem
Oracle-velocity EKF RMSE (0.02–0.02m, diagnostic) sits right next to AI-EKF
RMSE (**0.53m, drift 0.14%**, session 1 blackout — A) once GPS ground truth
itself is corrected. Velocity accuracy was never the bottleneck; a corrupted
GPS reference signal was. Fixing that reference (not the model) took drift
from 212–252% to well under the 10% target in one change.

## (b) Diagnosis
With velocity solved, heading is the next compounding error term. Gyro-only
integrated heading drifts against true course-over-ground with no correction
signal in this dataset denser than ~8.5s (real GPS fixes land roughly every
85 samples, not every sample) — same root pattern as the position bug.

## (c) This project's three contributions
1. **GPS reference fix (A):** corrupted stale-fix-then-jump reconstruction →
   coord/speed path-length ratio 3.25x → 0.998. AI-EKF drift: **0.14%**
   (target <10%).
2. **NHC + adaptive-R fusion (B):** implemented and tested on top of (1).
   Result: **no measurable improvement** (baseline 1.32m → NHC 1.45m,
   adaptive-R 1.32m, combined 1.50m). Honest finding: once the GPS
   reference is fixed, residual error is too small for these techniques to
   improve on, and NHC's zero-lateral-velocity assumption fights genuine
   cornering.
3. **Learned heading correction (C):** LSTM trained on genuine GPS-fix
   transitions only (860 labeled windows, 2 sessions) to predict
   gyro-vs-course residual. Standalone: 99-104° RMSE vs 114.78° no-correction
   baseline — modest. **Tested inside the actual EKF pipeline: made results
   worse (0.53m → 58.41m RMSE)**, so it was NOT adopted in the final
   pipeline. Root cause: too few real GPS fixes to train dense, low-noise
   labels; the model's ~100° error is too large to safely inject into
   heading. Kept as a tested-and-rejected finding, not hidden.

## Three-tier framing
| Tier | Approach | Result |
|---|---|---|
| Naive | Raw INS, no correction | 58.80m RMSE, 15.8% drift |
| Classical fusion | EKF, fixed-gain, classical velocity | 52.85m RMSE, 14.2% drift |
| **This project** | EKF + AI velocity + GPS-fix (heading correction tested, not adopted) | **0.53m RMSE, 0.14% drift** |
| Oracle ceiling (diagnostic) | True speed, true GPS | 0.02m RMSE — near-tautological once GPS itself is speed-derived; shown for reference, not as validation |

## (d) Honest limitations
- Single session trained + one held-out session (leave-one-out velocity
  RMSE: **1.67–1.69 m/s**, notably worse than same-session 0.17–0.33 m/s) —
  this is a real generalization gap, not solved, named as future work:
  more sessions, more devices, more drivers.
- Heading-correction model has very few genuine labels per session (74–786,
  imbalanced across sessions) because real GPS fixes are sparse; 99°
  residual RMSE is an improvement, not a solution.
- NHC/adaptive-R added no value here — worth stating plainly rather than
  cutting from the deck, since a judge asking "what else did you try" is a
  near-certainty.
- Position and RMSE both ultimately trace back to GPS-speed as the trusted
  reference signal; if that signal is itself imperfect, the whole eval
  chain inherits it. Independent ground truth (e.g. RTK log) is the next
  validation step, not yet available.
