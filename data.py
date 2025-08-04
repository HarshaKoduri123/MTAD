import torch
from torch.utils.data import Dataset

class VesselSequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        sequences: list or numpy array of shape (num_sequences, seq_length, num_features)
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return a single sequence as a tensor
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float32)