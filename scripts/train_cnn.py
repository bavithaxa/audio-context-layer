import json
import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = "data/processed/window_index.json"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

BEST_MODEL_PATH = MODEL_DIR / "audio_cnn_best.pt"
LOSS_CURVE_PATH = RESULTS_DIR / "training_loss.png"

SAMPLE_RATE = 16000

WINDOW_SECONDS = 5.0

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 0.001

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DATASET
# ============================================================

class AudioWindowDataset(Dataset):

    def __init__(
        self,
        windows,
        label_count
    ):

        self.windows = windows
        self.label_count = label_count

    def __len__(self):

        return len(self.windows)

    def __getitem__(self, index):

        item = self.windows[index]

        audio_path = item["audio_path"]

        start = item["start"]

        # ----------------------------------------------------
        # Load exactly 5 seconds
        # ----------------------------------------------------

        audio, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
            offset=start,
            duration=WINDOW_SECONDS
        )

        expected_length = int(
            SAMPLE_RATE * WINDOW_SECONDS
        )

        # ----------------------------------------------------
        # Pad shorter final windows
        # ----------------------------------------------------

        if len(audio) < expected_length:

            audio = np.pad(
                audio,
                (
                    0,
                    expected_length - len(audio)
                )
            )

        else:

            audio = audio[
                :expected_length
            ]

        # ----------------------------------------------------
        # Log Mel Spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # ----------------------------------------------------
        # Normalize per sample
        # ----------------------------------------------------

        mean = log_mel.mean()

        std = log_mel.std() + 1e-6

        log_mel = (
            log_mel - mean
        ) / std

        # ----------------------------------------------------
        # Convert:
        #
        # [mel, time]
        #
        # →
        #
        # [channel, mel, time]
        # ----------------------------------------------------

        features = torch.tensor(
            log_mel,
            dtype=torch.float32
        ).unsqueeze(0)

        target = torch.tensor(
            item["target"],
            dtype=torch.float32
        )

        return features, target


# ============================================================
# CNN MODEL
# ============================================================

class AudioCNN(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    micro_f1 = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    return micro_f1, macro_f1


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_targets = []
    all_probabilities = []

    with torch.no_grad():

        for features, targets in loader:

            features = features.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                targets
            )

            total_loss += (
                loss.item()
                * features.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_targets.append(
                targets.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    total_samples = len(
        loader.dataset
    )

    average_loss = (
        total_loss / total_samples
    )

    all_targets = np.concatenate(
        all_targets
    )

    all_probabilities = np.concatenate(
        all_probabilities
    )

    micro_f1, macro_f1 = (
        calculate_metrics(
            all_targets,
            all_probabilities
        )
    )

    return (
        average_loss,
        micro_f1,
        macro_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO CONTEXT LAYER")
    print("CNN AUDIO PERCEPTION MODEL")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------

    print("\nLoading training index...")

    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    labels = data["labels"]

    num_classes = len(labels)

    windows = data["windows"]

    print(
        f"Classes: {num_classes}"
    )

    print(
        f"Total windows: {len(windows)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_windows = [
        x
        for x in windows
        if x["split"] == "train"
    ]

    validation_windows = [
        x
        for x in windows
        if x["split"] == "validation"
    ]

    test_windows = [
        x
        for x in windows
        if x["split"] == "test"
    ]

    print(
        f"\nTrain windows: "
        f"{len(train_windows)}"
    )

    print(
        f"Validation windows: "
        f"{len(validation_windows)}"
    )

    print(
        f"Test windows: "
        f"{len(test_windows)}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("\nCreating datasets...")

    train_dataset = AudioWindowDataset(
        train_windows,
        num_classes
    )

    validation_dataset = AudioWindowDataset(
        validation_windows,
        num_classes
    )

    test_dataset = AudioWindowDataset(
        test_windows,
        num_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Calculate class weights
    # --------------------------------------------------------

    print("\nCalculating class weights...")

    train_targets = np.array([
        x["target"]
        for x in train_windows
    ])

    positive_counts = (
        train_targets.sum(axis=0)
    )

    negative_counts = (
        len(train_targets)
        - positive_counts
    )

    pos_weight = (
        negative_counts
        / np.maximum(
            positive_counts,
            1
        )
    )

    pos_weight_tensor = torch.tensor(
        pos_weight,
        dtype=torch.float32
    ).to(DEVICE)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating CNN...")

    model = AudioCNN(
        num_classes
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_losses = []
    validation_losses = []

    best_validation_f1 = -1.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n")
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        for features, targets in train_loader:

            features = features.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                features
            )

            loss = criterion(
                logits,
                targets
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * features.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        (
            validation_loss,
            validation_micro_f1,
            validation_macro_f1
        ) = evaluate(
            model,
            validation_loader,
            criterion
        )

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Micro F1: "
            f"{validation_micro_f1:.4f} | "
            f"Val Macro F1: "
            f"{validation_macro_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_micro_f1 > best_validation_f1:

            best_validation_f1 = (
                validation_micro_f1
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "labels":
                        labels,

                    "sample_rate":
                        SAMPLE_RATE,

                    "window_seconds":
                        WINDOW_SECONDS,

                    "n_mels":
                        N_MELS,

                    "n_fft":
                        N_FFT,

                    "hop_length":
                        HOP_LENGTH
                },
                BEST_MODEL_PATH
            )

            print(
                "  -> Best model saved."
            )

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(
            1,
            EPOCHS + 1
        ),
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        range(
            1,
            EPOCHS + 1
        ),
        validation_losses,
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "CNN Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        LOSS_CURVE_PATH,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\nLoading best model...")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    (
        final_val_loss,
        final_val_micro_f1,
        final_val_macro_f1
    ) = evaluate(
        model,
        validation_loader,
        criterion
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    (
        test_loss,
        test_micro_f1,
        test_macro_f1
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"\nValidation Loss: "
        f"{final_val_loss:.4f}"
    )

    print(
        f"Validation Micro F1: "
        f"{final_val_micro_f1:.4f}"
    )

    print(
        f"Validation Macro F1: "
        f"{final_val_macro_f1:.4f}"
    )

    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Micro F1: "
        f"{test_micro_f1:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_macro_f1:.4f}"
    )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nLoss curve:"
    )

    print(
        LOSS_CURVE_PATH
    )


if __name__ == "__main__":
    main()