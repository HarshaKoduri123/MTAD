import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from mtad import MultiModalAutoencoder
from loss import BFLoss
from data import VesselSequenceDataset
from utils import generate_vessel_sequences, normalize_sequences, compute_deltas

from config import (
    DATA, SAVE_DIR, SEQ_LENGTH, SEQ_LEN_MODEL, FEATURES, FEATURE_DIM,
    D_MODEL, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_INTERVAL,
    MODALITY_INDICES, LOSS_WEIGHTS, get_logger
)

logger = get_logger(__name__)


def save_checkpoint(model, epoch):
    checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint saved at: {checkpoint_path}")




def train_loop(model, dataloader, loss_fn, optimizer, device):
    epoch_losses = []
    epoch_re_errors = []
    epoch_mahalanobis = []


    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, total_re = 0.0, 0.0
        latent_vectors = []
        num_batches = 0
        

        for batch_sequences in dataloader:

        
            batch_sequences = batch_sequences.to(device)
            optimizer.zero_grad()

            embedded = model.embedding(batch_sequences)
            encoded = model.transformer_encoder(embedded)
            compressed = model.bottleneck(encoded)
            outputs = model.decoder(compressed)

            loss = loss_fn(outputs, batch_sequences)
            loss.backward()
            optimizer.step()
            

            re = F.mse_loss(outputs, batch_sequences, reduction='mean')
            batch_latents = encoded.detach().cpu().numpy().reshape(-1, D_MODEL)
            latent_vectors.append(batch_latents)

            total_loss += loss.item()
            total_re += re.item()
            num_batches += 1

        epoch_latents = np.concatenate(latent_vectors, axis=0)
        mu = np.mean(epoch_latents, axis=0)
        cov = np.cov(epoch_latents, rowvar=False) + 1e-6 * np.eye(D_MODEL)
        inv_cov = np.linalg.inv(cov)
        md = np.mean([np.sqrt((z - mu).T @ inv_cov @ (z - mu)) for z in epoch_latents])

        avg_loss = total_loss / num_batches
        avg_re = total_re / num_batches
        avg_md = md

        epoch_losses.append(avg_loss)
        epoch_re_errors.append(avg_re)
        epoch_mahalanobis.append(avg_md)

        logger.info(f"Epoch [{epoch}/{NUM_EPOCHS}] - "
                    f"Loss: {avg_loss:.6f} | RE: {avg_re:.6f} | MD: {avg_md:.6f}")

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, epoch)

    return epoch_losses, epoch_re_errors, epoch_mahalanobis


def main():


    logger.info("Loading data...")
    data = pd.read_csv(DATA)

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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = MultiModalAutoencoder(d_model=D_MODEL, feature_dim=FEATURE_DIM, seq_len=SEQ_LEN_MODEL).to(device)
    loss_fn = BFLoss(MODALITY_INDICES, LOSS_WEIGHTS).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting training loop...")
    train_loop(model, dataloader, loss_fn, optimizer, device)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
