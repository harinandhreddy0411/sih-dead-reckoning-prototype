import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

ACCELEROMETER_COLUMNS = ["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]
GRAVITY_COLUMNS = ["GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)"]
GYROSCOPE_COLUMNS = ["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]
FUSED_HEADING_DEGREES_COLUMN = "ORIENTATION (Yaw) (deg)"
GPS_LATITUDE_COLUMN = "GPS LATITUDE (degrees)"
GPS_LONGITUDE_COLUMN = "GPS LONGITUDE (degrees)"
GPS_SPEED_KMH_COLUMN = "GPS SPEED (Kmh)"

WINDOW_LENGTH_SAMPLES = 50
SAMPLE_RATE_HZ = 10.0
EARTH_RADIUS_METERS = 6371000.0
KMH_TO_MS = 1.0 / 3.6

LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
TRAINING_BATCH_SIZE = 64
TRAINING_LEARNING_RATE = 1e-3
TRAINING_EPOCHS = 15
MODEL_CHECKPOINT_PATH = "session1_velocity_model.pt"

VTA16_PATH = r"Synchronised V abd S datasets\Categorised IOVNB Dataset\Vta (Driver E)\Vta16\S-Vta16.csv"
SESSION_PATHS = ["data/S-Vta16.csv", "data/S-Vta2.csv"]


def convertGpsTrackToLocalEnuMeters(latitudeDegrees, longitudeDegrees):
    referenceLatitudeRadians = np.radians(latitudeDegrees[0])
    referenceLongitudeRadians = np.radians(longitudeDegrees[0])
    latitudeRadians = np.radians(latitudeDegrees)
    longitudeRadians = np.radians(longitudeDegrees)
    eastMeters = (longitudeRadians - referenceLongitudeRadians) * np.cos(referenceLatitudeRadians) * EARTH_RADIUS_METERS
    northMeters = (latitudeRadians - referenceLatitudeRadians) * EARTH_RADIUS_METERS
    return eastMeters, northMeters


def loadSequenceFromCsv(csvFilePath):
    rawDataFrame = pd.read_csv(csvFilePath, encoding="latin-1")
    rawDataFrame.columns = rawDataFrame.columns.str.strip()
    rawDataFrame = rawDataFrame.rename(columns={"ORIENTATION (Yaw) (Â°)": "ORIENTATION (Yaw) (deg)"})

    rawAccelerometer = rawDataFrame[ACCELEROMETER_COLUMNS].to_numpy(dtype=np.float32)
    gravityEstimate = rawDataFrame[GRAVITY_COLUMNS].to_numpy(dtype=np.float32)
    linearAcceleration = rawAccelerometer - gravityEstimate

    gyroscopeReadings = rawDataFrame[GYROSCOPE_COLUMNS].to_numpy(dtype=np.float32)
    fusedHeadingDegrees = rawDataFrame[FUSED_HEADING_DEGREES_COLUMN].to_numpy(dtype=np.float32)
    groundTruthSpeedMetersPerSecond = rawDataFrame[GPS_SPEED_KMH_COLUMN].to_numpy(dtype=np.float32) * KMH_TO_MS

    eastMeters, northMeters = convertGpsTrackToLocalEnuMeters(
        rawDataFrame[GPS_LATITUDE_COLUMN].to_numpy(dtype=np.float64),
        rawDataFrame[GPS_LONGITUDE_COLUMN].to_numpy(dtype=np.float64))

    return {
        "linearAcceleration": linearAcceleration,
        "gyroscopeReadings": gyroscopeReadings,
        "fusedHeadingDegrees": fusedHeadingDegrees,
        "groundTruthSpeedMetersPerSecond": groundTruthSpeedMetersPerSecond,
        "eastMeters": eastMeters.astype(np.float32),
        "northMeters": northMeters.astype(np.float32),
    }


class ImuSpeedWindowDataset(Dataset):
    def __init__(self, sequenceData, windowLength):
        headingRadians = np.radians(sequenceData["fusedHeadingDegrees"])
        headingSin = np.sin(headingRadians).astype(np.float32).reshape(-1, 1)
        headingCos = np.cos(headingRadians).astype(np.float32).reshape(-1, 1)
        self.imuFeatures = np.concatenate([
            sequenceData["linearAcceleration"], sequenceData["gyroscopeReadings"], headingSin, headingCos], axis=1)
        self.groundTruthSpeed = sequenceData["groundTruthSpeedMetersPerSecond"]
        self.windowLength = windowLength

    def __len__(self):
        return len(self.imuFeatures) - self.windowLength

    def __getitem__(self, windowStartIndex):
        windowEndIndex = windowStartIndex + self.windowLength
        imuWindow = self.imuFeatures[windowStartIndex:windowEndIndex]
        targetSpeed = self.groundTruthSpeed[windowEndIndex]
        return torch.from_numpy(imuWindow), torch.tensor([targetSpeed], dtype=torch.float32)


class SpeedCorrectionLstm(nn.Module):
    def __init__(self, inputFeatureCount, hiddenSize, numLayers):
        super().__init__()
        self.recurrentLayer = nn.LSTM(input_size=inputFeatureCount, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
        self.speedOutputLayer = nn.Linear(hiddenSize, 1)

    def forward(self, imuWindowBatch):
        recurrentOutput, (finalHiddenState, _) = self.recurrentLayer(imuWindowBatch)
        return self.speedOutputLayer(finalHiddenState[-1])


def evaluateBlackoutReconstructionFromData(speedModel, sequenceData, blackoutDurationSeconds=60):
    windowDataset = ImuSpeedWindowDataset(sequenceData, WINDOW_LENGTH_SAMPLES)
    blackoutStartIndex = 0
    blackoutSampleCount = int(blackoutDurationSeconds * SAMPLE_RATE_HZ)
    blackoutEndIndex = min(blackoutStartIndex + blackoutSampleCount, len(windowDataset))

    speedModel.eval()
    reconstructedEast = [sequenceData["eastMeters"][blackoutStartIndex]]
    reconstructedNorth = [sequenceData["northMeters"][blackoutStartIndex]]

    with torch.no_grad():
        for sampleIndex in range(blackoutStartIndex, blackoutEndIndex):
            imuWindow, _ = windowDataset[sampleIndex]
            predictedSpeed = speedModel(imuWindow.unsqueeze(0)).item()
            headingRadians = np.radians(sequenceData["fusedHeadingDegrees"][sampleIndex])
            stepDurationSeconds = 1.0 / SAMPLE_RATE_HZ
            reconstructedEast.append(reconstructedEast[-1] + predictedSpeed * np.sin(headingRadians) * stepDurationSeconds)
            reconstructedNorth.append(reconstructedNorth[-1] + predictedSpeed * np.cos(headingRadians) * stepDurationSeconds)

    groundTruthEast = sequenceData["eastMeters"][blackoutStartIndex:blackoutEndIndex]
    groundTruthNorth = sequenceData["northMeters"][blackoutStartIndex:blackoutEndIndex]

    plt.figure(figsize=(8, 8))
    plt.plot(groundTruthEast, groundTruthNorth, label="GPS ground truth", linewidth=2)
    plt.plot(reconstructedEast, reconstructedNorth, label="Model reconstruction during blackout", linestyle="--")
    plt.xlabel("East (meters)")
    plt.ylabel("North (meters)")
    plt.title(f"Simulated {blackoutDurationSeconds}s GNSS blackout reconstruction")
    plt.legend()
    plt.axis("equal")
    plt.savefig("blackout_reconstruction.png", dpi=150)
    print("saved blackout_reconstruction.png")


def trainOnDataset(trainingDataset, epochs=TRAINING_EPOCHS):
    trainingDataLoader = DataLoader(trainingDataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)
    speedModel = SpeedCorrectionLstm(inputFeatureCount=8, hiddenSize=LSTM_HIDDEN_SIZE, numLayers=LSTM_NUM_LAYERS)
    optimizer = torch.optim.Adam(speedModel.parameters(), lr=TRAINING_LEARNING_RATE)
    lossFunction = nn.MSELoss()
    for epochIndex in range(epochs):
        speedModel.train()
        accumulatedEpochLoss = 0.0
        for imuWindowBatch, targetSpeedBatch in trainingDataLoader:
            optimizer.zero_grad()
            batchLoss = lossFunction(speedModel(imuWindowBatch), targetSpeedBatch)
            batchLoss.backward()
            optimizer.step()
            accumulatedEpochLoss += batchLoss.item()
        print(f"epoch {epochIndex + 1}/{epochs} mean_loss {accumulatedEpochLoss / len(trainingDataLoader):.4f}")
    return speedModel


def evaluateVelocityRmse(speedModel, sequenceData):
    dataset = ImuSpeedWindowDataset(sequenceData, WINDOW_LENGTH_SAMPLES)
    speedModel.eval()
    errs = []
    with torch.no_grad():
        for i in range(len(dataset)):
            imuWindow, target = dataset[i]
            pred = speedModel(imuWindow.unsqueeze(0)).item()
            errs.append(pred - target.item())
    return float(np.sqrt(np.mean(np.square(errs))))


def leaveOneSessionOut(sessionPaths):
    """Honest generalization check: train on all-but-one session, evaluate
    velocity RMSE on the held-out session. Does NOT touch the deployed
    checkpoint -- diagnostic only."""
    sequences = [loadSequenceFromCsv(p) for p in sessionPaths]
    print("\n=== Leave-one-session-out validation ===")
    for holdoutIdx, holdoutPath in enumerate(sessionPaths):
        trainSeqs = [s for i, s in enumerate(sequences) if i != holdoutIdx]
        trainDatasets = [ImuSpeedWindowDataset(s, WINDOW_LENGTH_SAMPLES) for s in trainSeqs]
        combinedTrain = torch.utils.data.ConcatDataset(trainDatasets)
        model = trainOnDataset(combinedTrain)
        rmse = evaluateVelocityRmse(model, sequences[holdoutIdx])
        print(f"held out {holdoutPath}: velocity RMSE = {rmse:.3f} m/s")


def trainFinalModelOnAllSessions(sessionPaths):
    """Deployed checkpoint: trained on the union of all available sessions."""
    sequences = [loadSequenceFromCsv(p) for p in sessionPaths]
    datasets = [ImuSpeedWindowDataset(s, WINDOW_LENGTH_SAMPLES) for s in sequences]
    combined = torch.utils.data.ConcatDataset(datasets)
    speedModel = trainOnDataset(combined)
    torch.save(speedModel.state_dict(), MODEL_CHECKPOINT_PATH)
    print(f"saved {MODEL_CHECKPOINT_PATH} (trained on {len(sessionPaths)} sessions combined)")
    evaluateBlackoutReconstructionFromData(speedModel, sequences[0])
    return speedModel


def trainSpeedModelSingleSession(csvFilePath, trainFraction=0.8):
    fullSequenceData = loadSequenceFromCsv(csvFilePath)
    totalSampleCount = len(fullSequenceData["eastMeters"])
    trainEndIndex = int(totalSampleCount * trainFraction)
    testStartIndex = trainEndIndex + WINDOW_LENGTH_SAMPLES

    trainingSequenceData = {key: value[:trainEndIndex] for key, value in fullSequenceData.items()}
    testSequenceData = {key: value[testStartIndex:] for key, value in fullSequenceData.items()}

    trainingDataset = ImuSpeedWindowDataset(trainingSequenceData, WINDOW_LENGTH_SAMPLES)
    trainingDataLoader = DataLoader(trainingDataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

    speedModel = SpeedCorrectionLstm(inputFeatureCount=8, hiddenSize=LSTM_HIDDEN_SIZE, numLayers=LSTM_NUM_LAYERS)
    optimizer = torch.optim.Adam(speedModel.parameters(), lr=TRAINING_LEARNING_RATE)
    lossFunction = nn.MSELoss()

    for epochIndex in range(TRAINING_EPOCHS):
        speedModel.train()
        accumulatedEpochLoss = 0.0
        for imuWindowBatch, targetSpeedBatch in trainingDataLoader:
            optimizer.zero_grad()
            predictedSpeedBatch = speedModel(imuWindowBatch)
            batchLoss = lossFunction(predictedSpeedBatch, targetSpeedBatch)
            batchLoss.backward()
            optimizer.step()
            accumulatedEpochLoss += batchLoss.item()
        print(f"epoch {epochIndex + 1}/{TRAINING_EPOCHS} mean_loss {accumulatedEpochLoss / len(trainingDataLoader):.4f}")

    torch.save(speedModel.state_dict(), MODEL_CHECKPOINT_PATH)
    print(f"saved {MODEL_CHECKPOINT_PATH}")
    print("WARNING: trained and tested on the same drive session, chronologically split.")

    evaluateBlackoutReconstructionFromData(speedModel, testSequenceData)
    return speedModel


if __name__ == "__main__":
    leaveOneSessionOut(SESSION_PATHS)
    trainFinalModelOnAllSessions(SESSION_PATHS)
