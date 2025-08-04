from sklearn.preprocessing import StandardScaler
import numpy as np

def normalize_sequences(sequences):
    """
    Normalizes a list of sequences using StandardScaler.
    
    Args:
        sequences (list of np.ndarray): List of sequences, each of shape (seq_len, num_features).
        
    Returns:
        normalized_sequences (list of np.ndarray): List of normalized sequences.
        scaler (StandardScaler): The fitted scaler object.
    """
    all_data = np.vstack(sequences)
    scaler = StandardScaler()
    scaler.fit(all_data)
    normalized_sequences = [scaler.transform(seq) for seq in sequences]
    
    return normalized_sequences, scaler



def generate_vessel_sequences(df, seq_length, features):
    """
    Splits AIS data into fixed-length non-overlapping sequences per vessel (MMSI).

    Args:
        df (pd.DataFrame): Input DataFrame containing AIS data with MMSI and required features.
        seq_length (int): Length of each sequence.
        features (list of str): List of features to include in each sequence.

    Returns:
        all_sequences (list of np.ndarray): List of sequences of shape (seq_length, num_features).
    """
    def create_sequences(group):
        data = group[features].values
        sequences = []
        for start_idx in range(0, len(data) - seq_length + 1, seq_length):
            seq = data[start_idx:start_idx + seq_length]
            sequences.append(seq)
        return sequences

    all_sequences = []
    for mmsi, vessel_group in df.groupby('MMSI'):
        vessel_sequences = create_sequences(vessel_group)
        all_sequences.extend(vessel_sequences)

    print(f"Total sequences created: {len(all_sequences)}")
    return all_sequences


def compute_deltas(df):
    """
    Compute DeltaTime, DeltaLat, and DeltaLon for each vessel based on consecutive AIS messages.

    Args:
        df (pd.DataFrame): Input AIS DataFrame with columns: 'MMSI', 'BaseDateTime', 'LAT', 'LON'

    Returns:
        pd.DataFrame: DataFrame with new delta features
    """
   
    grouped = df.groupby('MMSI')

    
    df = grouped.apply(
        lambda group: group.assign(
            DeltaTime = group['BaseDateTime'].diff().dt.total_seconds().fillna(0),
            DeltaLat  = group['LAT'].diff().fillna(0),
            DeltaLon  = group['LON'].diff().fillna(0)
        )
    )

    df = df.reset_index(drop=True)

    return df



