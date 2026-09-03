# Intelligent Dead Reckoning for Smartphones (SIH 2026 – PS 26168)

Prototype for ISRO/Dept. of Space problem statement 26168: AI/ML-assisted dead
reckoning for GNSS-denied smartphone navigation.

## Pipeline
Smartphone IMU → LSTM virtual velocity sensor → velocity pseudo-measurement → 
lightweight 4-state EKF → GNSS-denied position estimate.

- **Model**: single-layer LSTM (hidden_size=64) + linear output head, predicting
  scalar forward velocity from an 8-feature, 5-second (50-sample @ 10Hz) window:
  linear acceleration (x,y,z), gyro (yaw,pitch,roll), sin(heading), cos(heading).
- **Fusion**: 4-state EKF (east, north, v_east, v_north), constant-velocity model,
  AI velocity injected as a pseudo-measurement; GPS position correction disabled
  during a simulated 60s GNSS blackout.
- **Baseline**: naive forward-acceleration integration (not a validated INS),
  included for comparison only.

## Results (single session, IO-VNBD Vta16)

| Metric | Simple integration | AI velocity |
|---|---|---|
| RMSE (m/s) | 2.66 | **0.33** |
| MAE (m/s) | 2.30 | 0.25 |
| Correlation | -0.14 | 0.86 |

Trajectory RMSE over blackout: accel-DR 937.8m, classical EKF 827.9m, 
AI-EKF **788.2m** (oracle true-speed EKF: 789.6m — AI-EKF is within 1.5m of the 
theoretical best achievable by improving velocity alone).

**Caveat**: current evaluation is same-session / in-sample (train and blackout
window drawn from one recording), not held-out generalization. Treated as a
feasibility result, not a generalization claim.

## Files
- `train.py` — trains the LSTM on one session (80/20 chronological split)
- `pod_ab_merge.py` — runs inference + EKF fusion over the blackout, produces metrics/plots
- `session1_velocity_model.pt` — trained checkpoint
- `data/S-Vta16.csv` — IO-VNBD session used (see [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) for dataset details/license)
- `results/` — generated plots

## Limitations
Same-session evaluation only; single-session validation; reduced 4-state EKF;
relies on phone's pre-fused heading (no magnetometer/full attitude estimation);
unvalidated accelerometer axis sign in the baseline; no cross-session/device
validation yet.

## Future Work
Cross-session and cross-device validation, improved attitude estimation,
adaptive measurement covariance, ZUPT/NHC/map-matching, Android deployment.
