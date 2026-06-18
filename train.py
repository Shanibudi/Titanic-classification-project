"""
Train a PyTorch classification model for Titanic survival prediction.

The script downloads the Titanic competition files directly from Kaggle using
kagglehub when train.csv is not already available locally. It then performs
preprocessing, splits Kaggle train.csv into 80% train and 20% validation,
trains with early stopping on validation loss, and saves the trained model
checkpoint to disk.
"""

from __future__ import annotations
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple
import kagglehub
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Hyperparameters
# =========================
SEED = 42
DATA_DIR = Path("data")
MODEL_PATH = Path("model.pt")
COMPETITION_NAME = "titanic"
VALIDATION_SIZE = 0.20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-5
HIDDEN_DIM_1 = 64
DROPOUT = 0.10
THRESHOLD = 0.5

# Features used by the model
NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "CabinKnown",
    "TicketGroupSize",
]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
TARGET_COLUMN = "Survived"


class TitanicMLP(nn.Module):
    """Small MLP model for binary classification."""

    def __init__(self, input_dim: int, dropout: float = DROPOUT) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM_1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_titanic_data(data_dir: Path = DATA_DIR) -> None:
    """Download Titanic competition data using kagglehub."""
    data_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    # Skip download if the main dataset files already exist
    if train_csv.exists() and test_csv.exists():
        print("Titanic data already exists. Skipping Kaggle download.")
        return

    print("Downloading Titanic data from Kaggle using kagglehub...")

    try:
        downloaded_path = Path(kagglehub.competition_download(COMPETITION_NAME))
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Make sure Kaggle authentication is configured "
            "and that the Titanic competition files are accessible from your Kaggle account."
        ) from exc

    # Copy the Kaggle files into the local data directory
    for file_name in ["train.csv", "test.csv", "gender_submission.csv"]:
        source = downloaded_path / file_name
        destination = data_dir / file_name

        if source.exists():
            shutil.copy2(source, destination)

    # Validate that the required files exist
    if not train_csv.exists():
        raise FileNotFoundError(
            "Expected data/train.csv after Kaggle download, but it was not found."
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            "Expected data/test.csv after Kaggle download, but it was not found."
        )

    print(f"Titanic data saved to: {data_dir.resolve()}")


def extract_title(name: str) -> str:
    """Extract passenger title from the Name column."""
    if pd.isna(name) or "," not in str(name) or "." not in str(name):
        return "Unknown"

    title = str(name).split(",", 1)[1].split(".", 1)[0].strip()
    common_titles = {"Mr", "Mrs", "Miss", "Master"}

    return title if title in common_titles else "Rare"


def get_column_or_default(df: pd.DataFrame, column: str, default_value) -> pd.Series:
    """Handle missing columns when the input CSV does not fully match the Kaggle Titanic format."""
    if column in df.columns:
        return df[column]

    return pd.Series(default_value, index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features used by the model."""
    df = df.copy()

    # Create title from the passenger name
    df["Title"] = get_column_or_default(df, "Name", "Unknown").apply(extract_title)

    # Create family related features
    sibsp = get_column_or_default(df, "SibSp", 0).fillna(0)
    parch = get_column_or_default(df, "Parch", 0).fillna(0)

    df["FamilySize"] = sibsp + parch + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Encode whether cabin information exists
    df["CabinKnown"] = get_column_or_default(df, "Cabin", np.nan).notna().astype(int)

    # Count how many passengers share the same ticket
    if "Ticket" in df.columns:
        df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    else:
        df["TicketGroupSize"] = 1

    # Ensure all expected features exist before preprocessing
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""

    # Numeric features are imputed and scaled
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical features are imputed and one hot encoded
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Convert numpy arrays into a PyTorch DataLoader."""
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate the model using validation loss, accuracy, and ROC AUC."""
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor).item()

        # Convert logits to survival probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y, predictions)
    roc_auc = roc_auc_score(y, probabilities)

    return loss, accuracy, roc_auc


def train_model() -> Dict[str, float]:
    """Train the Titanic survival prediction model."""
    set_seed(SEED)

    download_titanic_data(DATA_DIR)

    df = pd.read_csv(DATA_DIR / "train.csv")

    if TARGET_COLUMN not in df.columns:
        raise ValueError("train.csv must contain the Survived target column.")

    # Separate target from input features
    y = df[TARGET_COLUMN].astype(np.float32).values
    x_raw = add_features(df)

    # Split Kaggle train.csv into train and validation sets
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_raw,
        y,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Fit preprocessing only on the training split
    preprocessor = build_preprocessor()

    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float32)
    x_val = preprocessor.transform(x_val_raw).astype(np.float32)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"Input dimension after preprocessing: {x_train.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TitanicMLP(input_dim=x_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        # Training loop over mini batches
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Evaluate after each epoch on the validation split
        val_loss, val_acc, val_auc = evaluate_model(
            model,
            x_val,
            y_val,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_acc:.4f} | "
            f"val_roc_auc={val_auc:.4f}"
        )

        # Save the best model according to validation loss
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Stop training if validation loss does not improve
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training failed because no best model state was saved.")

    # Restore the best validation-loss model
    model.load_state_dict(best_state)
    
    # Recompute final validation metrics for the selected best model
    val_loss, val_acc, val_auc = evaluate_model(
        model,
        x_val,
        y_val,
        criterion,
        device,
    )

    # Save model, preprocessing pipeline, metrics, and hyperparameters
    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": x_train.shape[1],
        "dropout": DROPOUT,
        "preprocessor": preprocessor,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": THRESHOLD,
        "best_epoch": best_epoch,
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
        "validation_roc_auc": val_auc,
        "hyperparameters": {
            "seed": SEED,
            "validation_size": VALIDATION_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "dropout": DROPOUT,
            "hidden_dim_1": HIDDEN_DIM_1
        },
    }

    torch.save(checkpoint, MODEL_PATH)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation ROC AUC: {val_auc:.4f}")
    print(f"Saved checkpoint to: {MODEL_PATH}")

    return {
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
        "validation_roc_auc": val_auc,
    }


if __name__ == "__main__":
    train_model()