import torch.nn as nn
import torch.nn.functional as F

class BFLoss(nn.Module):
    """
    Behavioral Fusion Loss (BFLoss): Weighted reconstruction loss over multiple behavioral modalities.
    """

    def __init__(self, modality_indices, modality_weights=None):
        super(BFLoss, self).__init__()
        self.modality_indices = modality_indices

        # Default weights
        if modality_weights is None:
            modality_weights = {'time': 1.0, 'spatial': 3.0, 'speed': 2.0, 'direction': 2.0}

        # Normalize to sum to 1
        total = sum(modality_weights.values())
        self.weights = {k: v / total for k, v in modality_weights.items()}

    def forward(self, predictions, targets):
        """
        predictions and targets: (batch_size, seq_len, feature_dim)
        """
        loss = 0.0

        for modality, indices in self.modality_indices.items():
            pred = predictions[:, :, indices]
            true = targets[:, :, indices]
            modality_loss = F.mse_loss(pred, true)
            loss += self.weights[modality] * modality_loss

        return loss