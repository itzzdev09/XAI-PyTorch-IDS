import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch

def load_dataset(data_dir, filename=None):
    """
    Loads dataset(s) from the given directory.
    - If filename is provided, loads only that CSV.
    - Otherwise, loads and concatenates all CSVs in the directory.
    """
    if filename:  # Load one dataset
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{filename} not found in {data_dir}")
        print(f"Loading {filename}...")
        return pd.read_csv(file_path)
    
    # Default: load all
    dataframes = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            print(f"Loading {file}...")
            df = pd.read_csv(os.path.join(data_dir, file))
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def preprocess_dataset(df):
    # Drop identifiers and non-numeric fields
    drop_cols = ["FLOW_ID", "PROTOCOL_MAP", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ANALYSIS_TIMESTAMP"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Handle categorical label column
    label_encoder = LabelEncoder()
    df["ALERT"] = df["ALERT"].astype(str)  # ensure labels are strings
    y = label_encoder.fit_transform(df["ALERT"])
    df = df.drop(columns=["ALERT"])  # remove target from features

    # Ensure only numeric columns remain
    X = df.select_dtypes(include=["int64", "float64"]).values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, label_encoder


def prepare_dataloaders(X, y, batch_size=64, test_size=0.2):
    """
    Splits into train/test and creates PyTorch DataLoaders.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
