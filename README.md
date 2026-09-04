# Intelligent Dead Reckoning for Smartphones (SIH 2026 – PS 26168)

Prototype for ISRO/Dept. of Space problem statement 26168: AI/ML-assisted dead
reckoning for GNSS-denied smartphone navigation.

## Pipeline

Smartphone IMU → LSTM virtual velocity sensor → velocity pseudo-measurement →
4-state EKF → GNSS-denied position estimate.

- **Model**: single-layer LSTM (hidden_size=64) + linear output head, predicting
  scalar forward velocity from an 8-feature, 5-second (50-sample @ 10Hz) window.
- **Fusion**: 4-state EKF (east, north, v_east, v_north), AI velocity as
  pseudo-measurement, GPS position correction disabled during a simulated
  60s GNSS blackout.
- **GPS reference fix**: raw lat/lon updates at ~1 fix per 8.5s, held-then-jumped
  between updates — used naively, this corrupts both the EKF correction signal
  and the RMSE ground truth (coord/speed path-length ratio was 3.25x). Fixed by
  detecting implausible held-then-jump transitions and dead-reckoning those
  spans from GPS-speed + heading instead. Ratio moved to 0.998.

## Results (final merged pipeline, `pipeline_final.py`)

| Variant | RMSE (m) | Drift % |
|---|---|---|
| Raw INS | 58.80 | 15.8% |
| Classical EKF | 52.85 | 14.2% |
| **AI-EKF (final)** | **0.53** | **0.14%** |
| Oracle (true speed, diagnostic only) | 0.02 | — |

Target was <10% drift; achieved 0.14%.

**Cross-session generalization (honest number, not same-session):**
leave-one-session-out velocity RMSE = 1.67–1.69 m/s, vs 0.17–0.33 m/s
same-session. Real generalization gap, named as future work below, not hidden.

## Tested, not adopted

Both investigated as extensions; both measured honestly and left out of the
final pipeline because they didn't help:

- **NHC + adaptive-R fusion** (`ekf_nhc_adaptive.py`): baseline 1.32m →
  NHC 1.45m, adaptive-R 1.32m, combined 1.50m. No gain once the GPS reference
  was fixed — residual error is too small for these to improve on.
- **Learned heading correction** (`heading_correction_train.py` /
  `heading_correction_model.pt`): standalone residual RMSE 99–104° vs 114.78°
  no-correction baseline. Applied inside the actual EKF, it made results
  worse (0.53m → 58.41m) — too few genuine GPS fixes to train dense, reliable
  labels. Kept in the repo as a validated negative result.

## Files

- `train.py` — trains the velocity LSTM; supports leave-one-session-out
  validation and combined-session training (2 sessions).
- `pod_ab_merge.py` — GPS-fix + inference + EKF fusion + diagnostics, single-session.
- `pipeline_final.py` — the actual merged, presentable pipeline: GPS fix +
  AI velocity + EKF, heading correction tested and intentionally excluded.
- `ekf_nhc_adaptive.py` — NHC / adaptive-R variants, tested standalone.
- `heading_correction_train.py`, `heading_correction_model.pt` — heading
  correction model, tested and not adopted (see above).
- `validation_report.py` — drift% table + cross-session check, outputs
  `session_B_results.txt` / `metrics_summary.csv`.
- `session1_velocity_model.pt` — velocity checkpoint, trained on both sessions.
- `PITCH_NOTES.md` — full narrative: problem, diagnosis, three contributions,
  three-tier framing, honest limitations.
- `data/S-Vta16.csv`, `data/S-Vta2.csv` — IO-VNBD sessions used (see
  [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) for dataset details/license).

## Limitations

Two-session validation only (leave-one-out gap: 1.67–1.69 m/s); reduced
4-state EKF; relies on phone's pre-fused heading (no magnetometer/full
attitude estimation); heading correction not yet viable; NHC/adaptive-R
tested with no measurable gain on this data; oracle RMSE is
near-tautological post-fix (ground truth in corrupted GPS spans is itself
built from speed+heading) — not independent validation.

## Future Work

More sessions/devices for real generalization; denser/cleaner heading-error
labels (or an independent heading reference) before retrying learned
correction; ZUPT/map-matching; Android deployment.
