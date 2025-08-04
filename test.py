import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from utils import generate_vessel_sequences, normalize_sequences, compute_deltas
from mtad import MultiModalAutoencoder

from data import VesselSequenceDataset
import os
import pandas as pd

from config import (
    VAL_DATA, SAVE_DIR, SEQ_LENGTH, SEQ_LEN_MODEL, FEATURES, FEATURE_DIM,
    D_MODEL, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_INTERVAL,
    MODALITY_INDICES, LOSS_WEIGHTS, get_logger
)

logger = get_logger(__name__)

def compute_reconstruction_errors(model, dataloader):
    model.eval()
    errors = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon = model(batch)
            loss = torch.mean((batch - recon) ** 2, dim=(1, 2))
            errors.extend(loss.cpu().numpy())

    return np.array(errors)

def main():
    logger.info("Loading data...")
    data = pd.read_csv(VAL_DATA)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Grouping Data...")
    data['BaseDateTime'] = pd.to_datetime(data['BaseDateTime'])
    data = data.sort_values(by=['MMSI', 'BaseDateTime']).reset_index(drop=True)


    logger.info("Computing Deltas...")
    data = compute_deltas(data)

    logger.info("Generating sequences...")
    sequences = generate_vessel_sequences(data, SEQ_LENGTH, FEATURES)
    normalized_sequences, scaler = normalize_sequences(sequences)
 
    dataset = VesselSequenceDataset(normalized_sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    model = MultiModalAutoencoder(d_model=D_MODEL, feature_dim=FEATURE_DIM, seq_len=SEQ_LEN_MODEL).to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_epoch_50.pth"), map_location=device))
    print("Model loaded.")

    
    errors = compute_reconstruction_errors(model, dataloader)
    print(f"Mean reconstruction error: {np.mean(errors):.4f}")

    # 7. Compute anomaly threshold
    threshold = np.mean(errors) + 3 * np.std(errors)
    print(f"Anomaly threshold (mean + 3×std): {threshold:.4f}")

    # 8. Flag anomalies
    anomaly_flags = errors > threshold
    print(f"Anomalies detected: {np.sum(anomaly_flags)} / {len(errors)}")

if __name__ == "__main__":
    main()
