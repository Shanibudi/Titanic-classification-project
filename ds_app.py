"""
Streamlit inference and evaluation app for the Titanic PyTorch model.

Run with:
python -m streamlit run ds_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


HIDDEN_DIM_1 = 64
DROPOUT = 0.10
TARGET_COLUMN = "Survived"


class TitanicMLP(nn.Module):
    """Small MLP model for binary classification."""

    def __init__(self, input_dim: int, dropout: float = DROPOUT) -> None:
        super().__init__()

        # The same architecture as used in train.py
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM_1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def safe_torch_load(path: Path) -> Dict:
    """Load a PyTorch checkpoint while supporting different PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_title(name: str) -> str:
    """Extract passenger title from the Name column."""
    if pd.isna(name):
        return "Unknown"

    if "," not in name or "." not in name:
        return "Unknown"

    title = name.split(",", 1)[1].split(".", 1)[0].strip()
    common_titles = {"Mr", "Mrs", "Miss", "Master"}

    return title if title in common_titles else "Rare"


def add_features(df: pd.DataFrame, numeric_features, categorical_features) -> pd.DataFrame:
    """Create the same features used during training."""
    df = df.copy()

    # Create title from the passenger name
    df["Title"] = df.get("Name", pd.Series(index=df.index, dtype=str)).apply(extract_title)

    # Create family related features
    df["FamilySize"] = df.get("SibSp", 0).fillna(0) + df.get("Parch", 0).fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Encode whether cabin information exists
    df["CabinKnown"] = df.get("Cabin", pd.Series(index=df.index)).notna().astype(int)

    # Count how many passengers share the same ticket
    if "Ticket" in df.columns:
        df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    else:
        df["TicketGroupSize"] = 1

    # Ensure all expected features exist before applying the saved preprocessor
    for col in numeric_features:
        if col not in df.columns:
            df[col] = np.nan

    for col in categorical_features:
        if col not in df.columns:
            df[col] = "Unknown"

    return df[numeric_features + categorical_features]


@st.cache_resource
def load_model(model_path: str) -> Tuple[TitanicMLP, Dict]:
    """Load the trained model checkpoint and rebuild the PyTorch model."""
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    checkpoint = safe_torch_load(path)

    # Recreate the model with the same input dimension saved during training
    model = TitanicMLP(
        input_dim=checkpoint["input_dim"],
        dropout=checkpoint.get("dropout", DROPOUT),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict(df: pd.DataFrame, model: TitanicMLP, checkpoint: Dict) -> pd.DataFrame:
    """Run preprocessing and model inference on a Titanic format CSV."""
    
    # Create the same input features used during training
    x_raw = add_features(
        df,
        checkpoint["numeric_features"],
        checkpoint["categorical_features"],
    )

    # Apply the saved preprocessing pipeline from train.py
    x = checkpoint["preprocessor"].transform(x_raw).astype(np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Convert model logits into survival probabilities
    with torch.no_grad():
        logits = model(x_tensor)
        probabilities = torch.sigmoid(logits).numpy()

    threshold = checkpoint.get("threshold", 0.5)
    predictions = (probabilities >= threshold).astype(int)

    # Add predictions to the original dataframe
    output = df.copy()
    output["SurvivalProbability"] = probabilities
    output["PredictedSurvived"] = predictions

    return output


def plot_probability_histogram(results: pd.DataFrame) -> plt.Figure:
    """Plot the distribution of predicted survival probabilities."""
    fig, ax = plt.subplots()
    ax.hist(results["SurvivalProbability"], bins=20)
    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Passenger count")
    ax.set_title("Prediction probability distribution")
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """Plot ROC curve when true labels are available."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle=":", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()

    return fig


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Titanic Survival Classifier",
        page_icon="🚢",
        layout="wide",
    )

    st.title("Welcome to the Titanic Survival Classifier")
    st.write(
        "Load the trained PyTorch model and run inference on a Titanic CSV file. "
        "If the CSV contains a Survived column, the app also reports Accuracy and ROC AUC."
    )

    # User inputs for model path and CSV path
    model_path = st.text_input("Path to trained model", value="model.pt")
    csv_path = st.text_input("Path to test CSV", value="data/test.csv")

    run_button = st.button("Run inference")

    if not run_button:
        return

    # Load model and CSV file
    try:
        model, checkpoint = load_model(model_path)
        df = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(str(exc))
        return

    # Run model inference
    try:
        results = predict(df, model, checkpoint)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    st.subheader("Model information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best epoch", checkpoint.get("best_epoch", "N/A"))
    col2.metric("Saved validation Accuracy", f"{checkpoint.get('validation_accuracy', float('nan')):.4f}")
    col3.metric("Saved validation ROC AUC", f"{checkpoint.get('validation_roc_auc', float('nan')):.4f}")

    # If labels exist, compute evaluation metrics
    if TARGET_COLUMN in results.columns:
        y_true = results[TARGET_COLUMN].astype(int).values
        y_prob = results["SurvivalProbability"].values
        y_pred = results["PredictedSurvived"].values

        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)

        st.subheader("Evaluation metrics")

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Accuracy", f"{accuracy:.4f}")
        metric_col2.metric("ROC AUC", f"{roc_auc:.4f}")

        st.subheader("Plots")
        st.pyplot(plot_roc_curve(y_true, y_prob))
        st.pyplot(plot_probability_histogram(results))

    else:
        st.info("The CSV does not contain Survived column, so only prediction probabilities will be shown.")
        st.pyplot(plot_probability_histogram(results))

    # Show a preview of predictions
    st.subheader("Prediction preview")

    preview_columns = [
        col
        for col in [
            "PassengerId",
            "Name",
            "Sex",
            "Age",
            "Pclass",
            TARGET_COLUMN,
            "SurvivalProbability",
            "PredictedSurvived",
        ]
        if col in results.columns
    ]

    st.dataframe(results[preview_columns].head(30), use_container_width=True)

    # Allow the user to download the predictions
    csv_bytes = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download predictions CSV",
        data=csv_bytes,
        file_name="titanic_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
