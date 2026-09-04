"""
Person A deliverable: validation_report.py
Outputs: session_B_results.txt, metrics_summary.csv

Uses POSITIONAL column indices (not names) because degree/superscript symbols
mojibake inconsistently across different Vta* CSV files even under the same
encoding fix. Column order (0-indexed), confirmed from S-Vta16.csv header:
 0 GPS LATITUDE            1 GPS LONGITUDE          2 GPS ALTITUDE
 3 GPS SPEED (Kmh)         4 GPS ACCURACY           5 GPS ORIENTATION
 6 GPS SATELLITES          7 TIME SINCE START (ms)  8 DATE
 9 ACCEL X                10 ACCEL Y                11 ACCEL Z
12 GRAVITY X               13 GRAVITY Y              14 GRAVITY Z
15 GYRO Yaw                16 GYRO Pitch             17 GYRO Roll
18 MAG X                   19 MAG Y                  20 MAG Z
21 ORIENTATION Yaw         22 ORIENTATION Pitch      23 ORIENTATION Roll

If a different session file has a different column count/order, this will
raise an IndexError immediately rather than silently reading the wrong column
— check the file's actual header if that happens.
"""

import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_CKPT = "session1_velocity_model.pt"
ORIGINAL_CSV = "data/S-Vta16.csv"
SECOND_SESSION_CSV = "data/S-Vta2.csv"   # <-- change filename if you copy a different session

COL_GPS_SPEED_KMH = 3
COL_TIMESTAMP_MS = 7
COL_ACCEL_X, COL_ACCEL_Y, COL_ACCEL_Z = 9, 10, 11
COL_GRAVITY_X, COL_GRAVITY_Y, COL_GRAVITY_Z = 12, 13, 14
COL_GYRO_YAW, COL_GYRO_PITCH, COL_GYRO_ROLL = 15, 16, 17
COL_ORIENT_YAW = 21

ASSUMED_DT_MS = 100.0
DT_TOLERANCE_PCT = 2.0
SEQ_LEN = 50
BLACKOUT_START_INDEX = 100
BLACKOUT_LEN = 600

# Updated from pod_ab_merge.py after fixing the corrupted-GPS-fix bug
# (reconstructGpsPositions: dead-reckon implausible fixes via speed+heading
# instead of trusting raw lat/lon). coord/speed ratio moved 3.25x -> 0.998.
ORIGINAL_RMSE = {
    "accel_dr":      58.80,
    "classical_ekf": 52.85,
    "ai_ekf":        0.54,
    "oracle_ekf":    0.02,   # diagnostic only — near-tautological, see caveat below
}
ORIGINAL_DIST_M = 371.7


def load_raw(csv_path):
    return pd.read_csv(csv_path, encoding="latin-1", header=0)


def build_features(df):
    vals = df.to_numpy()
    ax = vals[:, COL_ACCEL_X].astype(np.float64) - vals[:, COL_GRAVITY_X].astype(np.float64)
    ay = vals[:, COL_ACCEL_Y].astype(np.float64) - vals[:, COL_GRAVITY_Y].astype(np.float64)
    az = vals[:, COL_ACCEL_Z].astype(np.float64) - vals[:, COL_GRAVITY_Z].astype(np.float64)
    gyaw = vals[:, COL_GYRO_YAW].astype(np.float64)
    gpitch = vals[:, COL_GYRO_PITCH].astype(np.float64)
    groll = vals[:, COL_GYRO_ROLL].astype(np.float64)
    heading_rad = np.deg2rad(vals[:, COL_ORIENT_YAW].astype(np.float64))
    sin_h = np.sin(heading_rad)
    cos_h = np.cos(heading_rad)
    feats = np.stack([ax, ay, az, gyaw, gpitch, groll, sin_h, cos_h], axis=1).astype(np.float32)
    target = (vals[:, COL_GPS_SPEED_KMH].astype(np.float64) / 3.6).astype(np.float32)  # km/h -> m/s
    return feats, target


class VelocityLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=1):
        super().__init__()
        self.recurrentLayer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.speedOutputLayer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.recurrentLayer(x)
        return self.speedOutputLayer(out[:, -1, :])


def check_dt(csv_path):
    df = load_raw(csv_path)
    ts = df.to_numpy()[:, COL_TIMESTAMP_MS].astype(np.float64)
    diffs_ms = np.diff(ts)
    actual_dt = float(np.median(diffs_ms))
    deviation_pct = abs(actual_dt - ASSUMED_DT_MS) / ASSUMED_DT_MS * 100.0
    msg = f"DT_ACTUAL_MS = {actual_dt:.2f}  # assumed {ASSUMED_DT_MS}"
    if deviation_pct > DT_TOLERANCE_PCT:
        msg += f"  *** WARNING: deviates {deviation_pct:.2f}% from assumption ***"
    return actual_dt, msg


def compute_drift_rmse_based(rmse_m, distance_m):
    return rmse_m / distance_m * 100.0


def load_sequence_from_csv(csv_path):
    df = load_raw(csv_path)
    feats, target = build_features(df)
    X, y = [], []
    for i in range(SEQ_LEN, len(feats)):
        X.append(feats[i - SEQ_LEN:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def run_inference(model, X):
    with torch.no_grad():
        xb = torch.from_numpy(X).float()
        return model(xb).squeeze(-1).numpy()


def blackout_rmse(y_true, y_pred, start, length):
    end = min(start + length, len(y_true))
    return float(np.sqrt(np.mean((y_true[start:end] - y_pred[start:end]) ** 2)))


def load_model():
    model = VelocityLSTM()
    state = torch.load(MODEL_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    report_lines = []

    dt_val, dt_msg = check_dt(ORIGINAL_CSV)
    report_lines.append(dt_msg)
    report_lines.append("")

    report_lines.append("=== Drift % (blackout, session 1 / original) ===")
    report_lines.append(f"{'variant':<15}{'rmse_m':>10}{'drift_rmse_pct':>18}")
    drift_rows = []
    for variant, rmse in ORIGINAL_RMSE.items():
        drift = compute_drift_rmse_based(rmse, ORIGINAL_DIST_M)
        report_lines.append(f"{variant:<15}{rmse:>10.2f}{drift:>18.2f}")
        drift_rows.append(("session1", variant, rmse, drift))
    report_lines.append("")
    report_lines.append("NOTE: terminal-error-based drift % NOT computed — needs raw per-timestep")
    report_lines.append("position-error array from pod_ab_merge.py's EKF run, not just final RMSE.")
    report_lines.append("NOTE: oracle_ekf RMSE is near-tautological post-fix (ground truth in")
    report_lines.append("corrupted GPS spans is now itself built from speed+heading) — do not")
    report_lines.append("present it as independent validation; ai_ekf is the real result.")
    report_lines.append("")

    report_lines.append("=== Cross-session validation (no retraining) ===")
    try:
        model = load_model()
        X2, y2 = load_sequence_from_csv(SECOND_SESSION_CSV)
        pred2 = run_inference(model, X2)
        rmse_full = float(np.sqrt(np.mean((y2 - pred2) ** 2)))
        rmse_blackout = blackout_rmse(y2, pred2, BLACKOUT_START_INDEX, BLACKOUT_LEN)
        report_lines.append(f"second_session_csv = {SECOND_SESSION_CSV}")
        report_lines.append(f"full_session_velocity_rmse_mps = {rmse_full:.3f}")
        report_lines.append(f"blackout_window_velocity_rmse_mps = {rmse_blackout:.3f}")
        report_lines.append("session1_blackout_velocity_rmse_mps (established) = 0.17")
        drift_rows.append(("session2_unseen", "ai_velocity_rmse_mps", rmse_blackout, None))
    except FileNotFoundError as e:
        report_lines.append(f"SKIPPED: {e}")
        report_lines.append("Point SECOND_SESSION_CSV at an actual second Vta*/S-*.csv file.")
    except IndexError as e:
        report_lines.append(f"SKIPPED: column-index mismatch ({e}) — this session's CSV layout differs from S-Vta16's; check its header manually.")

    with open("session_B_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    with open("metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "session", "rmse_m", "drift_pct"])
        for session, variant, rmse, drift in drift_rows:
            writer.writerow([variant, session, rmse, drift if drift is not None else ""])

    print("\n".join(report_lines))
    print("\nWrote: session_B_results.txt, metrics_summary.csv")


if __name__ == "__main__":
    main()