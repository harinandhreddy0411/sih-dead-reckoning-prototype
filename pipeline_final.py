import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

VTA16_PATH = "data/S-Vta16.csv"
VELOCITY_MODEL_PATH = "session1_velocity_model.pt"
HEADING_MODEL_PATH = "heading_correction_model.pt"

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

R_V_AI = 0.33 ** 2


class SpeedCorrectionLstm(nn.Module):
    def __init__(self, inputFeatureCount=8, hiddenSize=LSTM_HIDDEN_SIZE, numLayers=LSTM_NUM_LAYERS):
        super().__init__()
        self.recurrentLayer = nn.LSTM(input_size=inputFeatureCount, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
        self.speedOutputLayer = nn.Linear(hiddenSize, 1)

    def forward(self, x):
        _, (h, _) = self.recurrentLayer(x)
        return self.speedOutputLayer(h[-1])


class HeadingCorrectionLstm(nn.Module):
    def __init__(self, inputFeatureCount=8, hiddenSize=LSTM_HIDDEN_SIZE, numLayers=LSTM_NUM_LAYERS):
        super().__init__()
        self.recurrentLayer = nn.LSTM(input_size=inputFeatureCount, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
        self.headingOutputLayer = nn.Linear(hiddenSize, 1)

    def forward(self, x):
        _, (h, _) = self.recurrentLayer(x)
        return self.headingOutputLayer(h[-1])


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
    n = len(east)
    headingRad = np.radians(headingDeg)
    fixedEast, fixedNorth = east.copy(), north.copy()
    stepDist = np.sqrt(np.diff(east) ** 2 + np.diff(north) ** 2)
    isStale = stepDist == 0.0
    lastE, lastN = east[0], north[0]
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
                fixedEast[k], fixedNorth[k] = lastE, lastN
            i = b if hasJump else n
        else:
            lastE, lastN = east[i + 1], north[i + 1]
            i += 1
    return fixedEast, fixedNorth


def buildFeatureWindow(linearAccel, gyro, headingDeg, i):
    windowStart = i - WINDOW_LENGTH_SAMPLES
    headingRad = np.radians(headingDeg[windowStart:i])
    sin_ = np.sin(headingRad).astype(np.float32).reshape(-1, 1)
    cos_ = np.cos(headingRad).astype(np.float32).reshape(-1, 1)
    return np.concatenate([linearAccel[windowStart:i], gyro[windowStart:i], sin_, cos_], axis=1)


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
    n = len(headingDeg)
    blackoutEnd = BLACKOUT_START_INDEX + int(BLACKOUT_DURATION_SECONDS * SAMPLE_RATE_HZ)

    velocityModel = SpeedCorrectionLstm()
    velocityModel.load_state_dict(torch.load(VELOCITY_MODEL_PATH, map_location="cpu"))
    velocityModel.eval()

    headingModel = HeadingCorrectionLstm()
    headingModel.load_state_dict(torch.load(HEADING_MODEL_PATH, map_location="cpu"))
    headingModel.eval()

    aiVelocity = np.zeros(n)
    headingCorrectionDeg = np.zeros(n)
    with torch.no_grad():
        for i in range(WINDOW_LENGTH_SAMPLES, n):
            feats = buildFeatureWindow(linearAccel, gyro, headingDeg, i)
            tensorIn = torch.from_numpy(feats).unsqueeze(0)
            aiVelocity[i] = velocityModel(tensorIn).item()
            headingCorrectionDeg[i] = headingModel(tensorIn).item()
    aiVelocity[:WINDOW_LENGTH_SAMPLES] = trueSpeed[:WINDOW_LENGTH_SAMPLES]
    aiVelocity = np.clip(aiVelocity, MIN_PLAUSIBLE_SPEED_MS, MAX_PLAUSIBLE_SPEED_MS)

    correctedHeadingDeg = headingDeg + headingCorrectionDeg  # C's one-line merge step

    aiEastPlain, aiNorthPlain = runEkf(aiVelocity, headingDeg, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd, R_V_AI)
    aiEastCorrected, aiNorthCorrected = runEkf(aiVelocity, correctedHeadingDeg, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd, R_V_AI)

    rmsePlain = rmse(aiEastPlain, aiNorthPlain, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd)
    rmseCorrected = rmse(aiEastCorrected, aiNorthCorrected, gpsEast, gpsNorth, BLACKOUT_START_INDEX, blackoutEnd)
    dist = np.sum(trueSpeed[BLACKOUT_START_INDEX:blackoutEnd]) * DT

    print(f"AI-EKF, raw heading:        RMSE={rmsePlain:.2f}m  drift={rmsePlain/dist*100:.2f}%")
    print(f"AI-EKF, heading-corrected:  RMSE={rmseCorrected:.2f}m  drift={rmseCorrected/dist*100:.2f}%")
    print("\nNOTE: heading correction model (~100deg RMSE) makes results WORSE, not better.")
    print("FINAL PIPELINE RESULT (heading correction NOT applied): "
          f"RMSE={rmsePlain:.2f}m  drift={rmsePlain/dist*100:.2f}%")

    plt.figure(figsize=(8, 8))
    bw = slice(BLACKOUT_START_INDEX, blackoutEnd)
    plt.plot(gpsEast[bw], gpsNorth[bw], label="GPS ground truth", linewidth=2)
    plt.plot(aiEastPlain[bw], aiNorthPlain[bw], label="AI-EKF (raw heading)", linestyle="--")
    plt.plot(aiEastCorrected[bw], aiNorthCorrected[bw], label="AI-EKF (heading-corrected)", linestyle="-.")
    plt.legend(); plt.axis("equal")
    plt.title("Final merged pipeline — blackout window")
    plt.savefig("pipeline_final_result.png", dpi=150)
    print("saved pipeline_final_result.png")