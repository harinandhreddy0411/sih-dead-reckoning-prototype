"""
Person C deliverable: heading_correction_train.py -> heading_correction_model.pt

Target = courseOverGround(GPS) - gyroIntegratedHeading, but ONLY at genuine
GPS fix transitions. This dataset's raw GPS updates every ~85 samples (same
pattern found in the position-fix bug); everywhere else, "course" would have
to be derived from already-dead-reckoned points, which is circular (near-zero
signal, not a real label). So each real fix-to-fix jump gives one training
label; the model is fed the 50-sample IMU window ending at that jump, same
windowing convention as the velocity model.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOW_LENGTH_SAMPLES = 50
SAMPLE_RATE_HZ = 10.0
DT = 1.0 / SAMPLE_RATE_HZ
EARTH_RADIUS_METERS = 6371000.0
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
TRAINING_EPOCHS = 30
TRAINING_BATCH_SIZE = 16
MAX_PLAUSIBLE_SPEED_MS = 15.0
MODEL_CHECKPOINT_PATH = "heading_correction_model.pt"
SESSION_PATHS = ["data/S-Vta16.csv", "data/S-Vta2.csv"]


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


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
    refLat, refLon = np.radians(lat[0]), np.radians(lon[0])
    east = (np.radians(lon) - refLon) * np.cos(refLat) * EARTH_RADIUS_METERS
    north = (np.radians(lat) - refLat) * EARTH_RADIUS_METERS
    return linearAccel, gyro, headingDeg, east, north


def findTrustedTransitions(east, north, maxSpeedMs=MAX_PLAUSIBLE_SPEED_MS):
    """Same detection as pod_ab_merge.py's fix, but returns (a, b) index pairs
    for the genuine, plausible fix-to-fix jumps only."""
    n = len(east)
    stepDist = np.sqrt(np.diff(east) ** 2 + np.diff(north) ** 2)
    isStale = stepDist == 0.0
    pairs = []
    i = 0
    while i < n - 1:
        if isStale[i]:
            a = i
            j = i
            while j < n - 1 and isStale[j]:
                j += 1
            if j < n - 1:
                runLen = max(j - a, 1)
                impliedSpeed = stepDist[j] / (runLen * DT)
                if impliedSpeed <= maxSpeedMs:
                    pairs.append((a, j + 1))
            i = j + 1
        else:
            pairs.append((i, i + 1))
            i += 1
    return pairs


def buildLabeledWindows(csvPath):
    linearAccel, gyro, headingDeg, east, north = loadSequence(csvPath)
    headingRad = np.radians(headingDeg)
    sin_ = np.sin(headingRad).reshape(-1, 1).astype(np.float32)
    cos_ = np.cos(headingRad).reshape(-1, 1).astype(np.float32)
    feats = np.concatenate([linearAccel, gyro, sin_, cos_], axis=1)

    pairs = findTrustedTransitions(east, north)
    X, y = [], []
    for a, b in pairs:
        if b < WINDOW_LENGTH_SAMPLES:
            continue
        course = np.degrees(np.arctan2(east[b] - east[a], north[b] - north[a]))
        gyroYawDeg = np.degrees(gyro[a:b, 0])
        gyroIntegHeading = headingDeg[a] + np.sum(gyroYawDeg) * DT
        residual = wrap180(course - gyroIntegHeading)
        X.append(feats[b - WINDOW_LENGTH_SAMPLES:b])
        y.append(residual)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor([self.y[i]], dtype=torch.float32)


class HeadingCorrectionLstm(nn.Module):
    def __init__(self, inputFeatureCount=8, hiddenSize=LSTM_HIDDEN_SIZE, numLayers=LSTM_NUM_LAYERS):
        super().__init__()
        self.recurrentLayer = nn.LSTM(input_size=inputFeatureCount, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
        self.headingOutputLayer = nn.Linear(hiddenSize, 1)

    def forward(self, imuWindowBatch):
        _, (finalHiddenState, _) = self.recurrentLayer(imuWindowBatch)
        return self.headingOutputLayer(finalHiddenState[-1])


def main():
    allX, allY = [], []
    for p in SESSION_PATHS:
        X, y = buildLabeledWindows(p)
        print(f"{p}: {len(X)} trusted-transition labels, residual mean={y.mean():.2f} std={y.std():.2f} deg")
        allX.append(X)
        allY.append(y)
    X = np.concatenate(allX)
    y = np.concatenate(allY)
    print(f"total labeled windows: {len(X)}")

    splitIdx = int(len(X) * 0.8)
    perm = np.random.RandomState(0).permutation(len(X))
    trainIdx, testIdx = perm[:splitIdx], perm[splitIdx:]

    trainLoader = DataLoader(WindowDataset(X[trainIdx], y[trainIdx]), batch_size=TRAINING_BATCH_SIZE, shuffle=True)
    model = HeadingCorrectionLstm()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossFn = nn.MSELoss()

    for epoch in range(TRAINING_EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in trainLoader:
            optimizer.zero_grad()
            loss = lossFn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1}/{TRAINING_EPOCHS} mean_loss {total/len(trainLoader):.3f}")

    model.eval()
    with torch.no_grad():
        predTest = model(torch.from_numpy(X[testIdx])).squeeze(-1).numpy()
    testRmse = float(np.sqrt(np.mean((predTest - y[testIdx]) ** 2)))
    baselineRmse = float(np.sqrt(np.mean((0 - y[testIdx]) ** 2)))  # "no correction" baseline
    print(f"\nheld-out heading-residual RMSE: model={testRmse:.2f} deg vs no-correction baseline={baselineRmse:.2f} deg")

    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    print(f"saved {MODEL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()