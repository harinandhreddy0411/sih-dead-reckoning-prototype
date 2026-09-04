import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

VTA16_PATH = "data/S-Vta16.csv"
MODEL_CHECKPOINT_PATH = "session1_velocity_model.pt"

SAMPLE_RATE_HZ = 10.0
DT = 1.0 / SAMPLE_RATE_HZ
EARTH_RADIUS_METERS = 6371000.0
WINDOW_LENGTH_SAMPLES = 50
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1

BLACKOUT_DURATION_SECONDS = 60
BLACKOUT_START_INDEX = 100

MAX_PLAUSIBLE_SPEED_MS = 15.0
MIN_PLAUSIBLE_SPEED_MS = 0.0

R_V_CLASSICAL = 2.66 ** 2   # = 7.0756 — measured classical-velocity RMSE^2 (m/s)^2
R_V_AI = 0.33 ** 2          # = 0.1089 — measured AI-velocity RMSE^2 (m/s)^2


class SpeedCorrectionLstm(nn.Module):
    def __init__(self, inputFeatureCount, hiddenSize, numLayers):
        super().__init__()
        self.recurrentLayer = nn.LSTM(input_size=inputFeatureCount, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
        self.speedOutputLayer = nn.Linear(hiddenSize, 1)

    def forward(self, imuWindowBatch):
        recurrentOutput, (finalHiddenState, _) = self.recurrentLayer(imuWindowBatch)
        return self.speedOutputLayer(finalHiddenState[-1])


def loadSequence(csvPath):
    df = pd.read_csv(csvPath, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"ORIENTATION (Yaw) (Â°)": "ORIENTATION (Yaw) (deg)"})

    accel = df[["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]].to_numpy(dtype=np.float32)
    gravity = df[["GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)"]].to_numpy(dtype=np.float32)
    linearAccel = accel - gravity
    gyro = df[["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]].to_numpy(dtype=np.float32)
    headingDeg = df["ORIENTATION (Yaw) (deg)"].to_numpy(dtype=np.float32)
    lat = df["GPS LATITUDE (degrees)"].to_numpy(dtype=np.float64)
    lon = df["GPS LONGITUDE (degrees)"].to_numpy(dtype=np.float64)
    speedKmh = df["GPS SPEED (Kmh)"].to_numpy(dtype=np.float32)

    refLat, refLon = np.radians(lat[0]), np.radians(lon[0])
    east = (np.radians(lon) - refLon) * np.cos(refLat) * EARTH_RADIUS_METERS
    north = (np.radians(lat) - refLat) * EARTH_RADIUS_METERS

    speed = speedKmh / 3.6
    east, north = reconstructGpsPositions(east, north, speed, headingDeg)
    return linearAccel, gyro, headingDeg, east.astype(np.float32), north.astype(np.float32), speed


def reconstructGpsPositions(east, north, speed, headingDeg, maxSpeedMs=MAX_PLAUSIBLE_SPEED_MS):
    """A held (stale) fix followed by a jump is only trustworthy if the jump
    distance is plausible GIVEN the actual hold duration (jumpDist / (runLen*DT)),
    not a fixed single-sample threshold. When implausible, the raw fix is
    corrupted (not just noisy) -- discard it and dead-reckon that span using
    the independent GPS-speed + heading signal from the last trusted fix,
    same treatment the EKF already gives a real blackout."""
    n = len(east)
    headingRad = np.radians(headingDeg)
    fixedEast, fixedNorth = east.copy(), north.copy()
    stepDist = np.sqrt(np.diff(east) ** 2 + np.diff(north) ** 2)
    isStale = stepDist == 0.0
    lastE, lastN = east[0], north[0]
    untrustedCount = 0
    i = 0
    while i < n - 1:
        if isStale[i]:
            a = i
            j = i
            while j < n - 1 and isStale[j]:
                j += 1
            hasJump = j < n - 1
            runLen = max(j - a, 1)
            impliedSpeed = (stepDist[j] / (runLen * DT)) if hasJump else 0.0
            trusted = hasJump and impliedSpeed <= maxSpeedMs
            b = j + 1 if hasJump else n - 1
            for k in range(a + 1, b + 1):
                if trusted and k == b:
                    lastE, lastN = east[k], north[k]
                else:
                    lastE = lastE + speed[k - 1] * np.sin(headingRad[k - 1]) * DT
                    lastN = lastN + speed[k - 1] * np.cos(headingRad[k - 1]) * DT
                    untrustedCount += 1
                fixedEast[k], fixedNorth[k] = lastE, lastN
            i = b if hasJump else n
        else:
            lastE, lastN = east[i + 1], north[i + 1]
            i += 1
    if untrustedCount:
        print(f"reconstructGpsPositions: dead-reckoned {untrustedCount} corrupted-fix sample(s) "
              f"using GPS-speed+heading instead of trusting raw lat/lon")
    return fixedEast, fixedNorth


def loadModel():
    model = SpeedCorrectionLstm(inputFeatureCount=8, hiddenSize=LSTM_HIDDEN_SIZE, numLayers=LSTM_NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def aiVelocityAtIndex(model, linearAccel, gyro, headingDeg, i):
    windowStart = i - WINDOW_LENGTH_SAMPLES
    accelWindow = linearAccel[windowStart:i]
    gyroWindow = gyro[windowStart:i]
    headingWindow = headingDeg[windowStart:i]

    headingRad = np.radians(headingWindow)
    sin_ = np.sin(headingRad).astype(np.float32).reshape(-1, 1)
    cos_ = np.cos(headingRad).astype(np.float32).reshape(-1, 1)
    features = np.concatenate([accelWindow, gyroWindow, sin_, cos_], axis=1)

    with torch.no_grad():
        tensorIn = torch.from_numpy(features).unsqueeze(0)
        return model(tensorIn).item()


def classicalVelocityFromAccel(linearAccel):
    forwardAccel = -linearAccel[:, 0]
    return np.cumsum(forwardAccel) * DT


def runRawInsDeadReckoning(velocity, headingDeg, startEast, startNorth):
    headingRad = np.radians(headingDeg)
    east = [startEast]
    north = [startNorth]
    for i in range(len(velocity)):
        east.append(east[-1] + velocity[i] * np.sin(headingRad[i]) * DT)
        north.append(north[-1] + velocity[i] * np.cos(headingRad[i]) * DT)
    return np.array(east[1:]), np.array(north[1:])


def runEkf(velocity, headingDeg, gpsEast, gpsNorth, blackoutStart, blackoutEnd, rV):
    headingRad = np.radians(headingDeg)
    n = len(velocity)

    x = np.array([gpsEast[0], gpsNorth[0], 0.0, 0.0])
    P = np.eye(4) * 5.0
    F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]])
    Q = np.diag([0.05, 0.05, 0.5, 0.5])
    H_v = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    R_v = np.eye(2) * rV
    H_p = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R_p = np.eye(2) * 5.0

    estEast, estNorth = [x[0]], [x[1]]

    for i in range(n):
        x = F @ x
        P = F @ P @ F.T + Q

        zv = np.array([velocity[i] * np.sin(headingRad[i]), velocity[i] * np.cos(headingRad[i])])
        yv = zv - H_v @ x
        Sv = H_v @ P @ H_v.T + R_v
        Kv = P @ H_v.T @ np.linalg.inv(Sv)
        x = x + Kv @ yv
        P = (np.eye(4) - Kv @ H_v) @ P

        if not (blackoutStart <= i < blackoutEnd):
            zp = np.array([gpsEast[i], gpsNorth[i]])
            yp = zp - H_p @ x
            Sp = H_p @ P @ H_p.T + R_p
            Kp = P @ H_p.T @ np.linalg.inv(Sp)
            x = x + Kp @ yp
            P = (np.eye(4) - Kp @ H_p) @ P

        estEast.append(x[0])
        estNorth.append(x[1])

    return np.array(estEast[1:]), np.array(estNorth[1:])


def rmse(estE, estN, gpsE, gpsN, i0, i1):
    e = estE[i0:i1] - gpsE[i0:i1]
    n_ = estN[i0:i1] - gpsN[i0:i1]
    return np.sqrt(np.mean(e**2 + n_**2))


if __name__ == "__main__":
    linearAccel, gyro, headingDeg, gpsEast, gpsNorth, trueSpeed = loadSequence(VTA16_PATH)
    model = loadModel()

    blackoutEnd = BLACKOUT_START_INDEX + int(BLACKOUT_DURATION_SECONDS * SAMPLE_RATE_HZ)
    n = len(headingDeg)

    correctedHeadingDeg = headingDeg  # raw dataset heading, no GPS-anchored correction (known-good)

    classicalVelocityRaw = classicalVelocityFromAccel(linearAccel)
    classicalVelocityForFusion = np.clip(classicalVelocityRaw, MIN_PLAUSIBLE_SPEED_MS, MAX_PLAUSIBLE_SPEED_MS)

    print("computing AI velocity per timestep (this loops, may take a moment)...")
    aiVelocity = np.zeros(n)
    for i in range(WINDOW_LENGTH_SAMPLES, n):
        aiVelocity[i] = aiVelocityAtIndex(model, linearAccel, gyro, headingDeg, i)
    aiVelocity[:WINDOW_LENGTH_SAMPLES] = classicalVelocityForFusion[:WINDOW_LENGTH_SAMPLES]
    aiVelocity = np.clip(aiVelocity, MIN_PLAUSIBLE_SPEED_MS, MAX_PLAUSIBLE_SPEED_MS)

    print("\n--- blackout window velocity comparison (m/s), first 10 samples ---")
    for i in range(BLACKOUT_START_INDEX, BLACKOUT_START_INDEX + 10):
        print(f"idx {i}: true={trueSpeed[i]:.2f}  classical={classicalVelocityForFusion[i]:.2f}  ai={aiVelocity[i]:.2f}")

    # ---- Full-window velocity diagnostics ----
    bwSlice = slice(BLACKOUT_START_INDEX, blackoutEnd)
    trueBw = trueSpeed[bwSlice]
    classicalBw = classicalVelocityForFusion[bwSlice]
    aiBw = aiVelocity[bwSlice]

    def fullStats(pred, true, label):
        err = pred - true
        rmseV = np.sqrt(np.mean(err**2))
        maeV = np.mean(np.abs(err))
        biasV = np.mean(err)
        corrV = np.corrcoef(pred, true)[0, 1] if np.std(pred) > 1e-6 else float('nan')
        distV = np.sum(pred) * DT
        print(f"{label}: RMSE={rmseV:.2f} MAE={maeV:.2f} bias={biasV:.2f} corr={corrV:.2f} "
              f"mean={pred.mean():.2f} min={pred.min():.2f} max={pred.max():.2f} integrated_dist={distV:.1f}m")

    print("\n--- Full 60s blackout velocity metrics ---")
    trueDist = np.sum(trueBw) * DT
    print(f"ground truth: mean={trueBw.mean():.2f} min={trueBw.min():.2f} max={trueBw.max():.2f} integrated_dist={trueDist:.1f}m")
    fullStats(classicalBw, trueBw, "classical")
    fullStats(aiBw, trueBw, "ai (clipped)")

    # raw unclipped AI check
    aiVelocityRawUnclipped = np.zeros(n)
    for i in range(WINDOW_LENGTH_SAMPLES, n):
        aiVelocityRawUnclipped[i] = aiVelocityAtIndex(model, linearAccel, gyro, headingDeg, i)
    aiBwRaw = aiVelocityRawUnclipped[bwSlice]
    print(f"\nAI raw UNCLIPPED in blackout: min={aiBwRaw.min():.2f} max={aiBwRaw.max():.2f} "
          f"mean={aiBwRaw.mean():.2f} any_negative={np.any(aiBwRaw < 0)} any_above_15={np.any(aiBwRaw > 15)}")

    plt.figure(figsize=(12, 5))
    tAxis = np.arange(BLACKOUT_START_INDEX, blackoutEnd)
    plt.plot(tAxis, trueBw, label="Ground truth speed", linewidth=2)
    plt.plot(tAxis, classicalBw, label="Classical velocity", linestyle="--")
    plt.plot(tAxis, aiBw, label="AI (LSTM) velocity", linestyle="-.")
    plt.xlabel("Sample index"); plt.ylabel("Speed (m/s)")
    plt.title("Full 60s blackout: ground truth vs classical vs AI velocity")
    plt.legend()
    plt.savefig("blackout_velocity_full.png", dpi=150)
    print("saved blackout_velocity_full.png")

    # ---- Final sanity check: spatial/temporal scale consistency ----
    gpsEastBw = gpsEast[bwSlice]
    gpsNorthBw = gpsNorth[bwSlice]

    coordStepDist = np.sqrt(np.diff(gpsEastBw)**2 + np.diff(gpsNorthBw)**2)
    coordPathLength = np.sum(coordStepDist)
    speedIntegratedDist = np.sum(trueBw) * DT
    ratio = coordPathLength / speedIntegratedDist if speedIntegratedDist > 0 else float('nan')

    impliedSpeed = coordStepDist / DT

    print("\n--- Final sanity check: spatial/temporal scale ---")
    print(f"1. GPS-coordinate path length: {coordPathLength:.2f} m")
    print(f"2. GPS-speed-integrated distance: {speedIntegratedDist:.2f} m")
    print(f"3. Ratio (coord/speed): {ratio:.3f}")
    print(f"4. Coordinate step distance per sample: mean={coordStepDist.mean():.3f} m, "
          f"median={np.median(coordStepDist):.3f} m, max={coordStepDist.max():.3f} m")
    print(f"5. Implied mean GPS speed from coordinate steps: {impliedSpeed.mean():.2f} m/s "
          f"(median={np.median(impliedSpeed):.2f}, max={impliedSpeed.max():.2f})")

    timeDeltasRaw = pd.read_csv(VTA16_PATH, encoding="latin-1")
    timeDeltasRaw.columns = timeDeltasRaw.columns.str.strip()
    tsCol = timeDeltasRaw["TIME SINCE START (ms)"].to_numpy(dtype=np.float64)
    dtCheck = np.diff(tsCol[BLACKOUT_START_INDEX:blackoutEnd + 1])
    print(f"6. Actual DT in blackout window from timestamps: mean={dtCheck.mean():.2f}ms, "
          f"median={np.median(dtCheck):.2f}ms, min={dtCheck.min():.2f}ms, max={dtCheck.max():.2f}ms "
          f"(expected 100.00ms if DT=0.1s is correct)")

    # oracle-speed diagnostic (NOT part of final system, diagnostic only)
    oracleEast, oracleNorth = runEkf(trueSpeed, correctedHeadingDeg, gpsEast, gpsNorth,
                                      BLACKOUT_START_INDEX, blackoutEnd, R_V_AI)
    oracleRmse = rmse(oracleEast, oracleNorth, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd)
    print(f"\n[DIAGNOSTIC ONLY, not for deck] Oracle-speed EKF RMSE: {oracleRmse:.2f} m")

    rawEast, rawNorth = runRawInsDeadReckoning(classicalVelocityRaw, correctedHeadingDeg, gpsEast[0], gpsNorth[0])
    classicalEast, classicalNorth = runEkf(classicalVelocityForFusion, correctedHeadingDeg, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd, R_V_CLASSICAL)
    aiEast, aiNorth = runEkf(aiVelocity, correctedHeadingDeg, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd, R_V_AI)

    print(f"\n{'variant':<24} {'blackout RMSE (m)':>18}")
    for name, (e, nn_) in [("raw INS", (rawEast, rawNorth)),
                            ("classical EKF", (classicalEast, classicalNorth)),
                            ("AI-assisted EKF", (aiEast, aiNorth))]:
        r = rmse(e, nn_, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd)
        print(f"{name:<24} {r:>18.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].plot(gpsEast, gpsNorth, label="GPS ground truth", linewidth=2)
    axes[0].plot(rawEast, rawNorth, label="Raw INS (no correction)", linestyle=":")
    axes[0].plot(classicalEast, classicalNorth, label="Classical EKF", linestyle="--")
    axes[0].plot(aiEast, aiNorth, label="AI-assisted EKF", linestyle="-.")
    axes[0].set_title("Full route")
    axes[0].set_xlabel("East (m)"); axes[0].set_ylabel("North (m)")
    axes[0].axis("equal"); axes[0].legend()

    bw = slice(BLACKOUT_START_INDEX, blackoutEnd)
    axes[1].plot(gpsEast[bw], gpsNorth[bw], label="GPS ground truth", linewidth=2)
    axes[1].plot(rawEast[bw], rawNorth[bw], label="Raw INS", linestyle=":")
    axes[1].plot(classicalEast[bw], classicalNorth[bw], label="Classical EKF", linestyle="--")
    axes[1].plot(aiEast[bw], aiNorth[bw], label="AI-assisted EKF", linestyle="-.")
    axes[1].set_title(f"Blackout window ({BLACKOUT_DURATION_SECONDS}s)")
    axes[1].set_xlabel("East (m)"); axes[1].set_ylabel("North (m)")
    axes[1].axis("equal"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("pod_ab_evidence_plot.png", dpi=150)
    print("\nsaved pod_ab_evidence_plot.png")
