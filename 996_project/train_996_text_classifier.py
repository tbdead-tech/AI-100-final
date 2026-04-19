import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Config
# ----------------------------
EXCEL_PATH = "996ICU reworked data.xlsx"
TEXT_COL = "wrongcolumnname"

# If you manually add labels later, put them in a column named 'label'
# with values 0, 1, 2
LABEL_COL = "label"

# Output files
LABELING_CSV = "to_label_996.csv"
MODEL_OUT = "model_996_text_classifier.h5"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Text vectorization
SEQ_LEN = 200         # max characters kept from each description
VOCAB_SIZE = 8000     # character vocab size

# Training
EPOCHS = 15
BATCH_SIZE = 512


# ----------------------------
# 1) Load + clean dataset
# ----------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Drop the markdown separator row like ":---:"
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Remove obvious junk rows
    df = df[df[TEXT_COL].notna()]
    df = df[df[TEXT_COL] != ":---:"]
    df = df[df[TEXT_COL].str.len() > 0].copy()

    # Normalize spaces
    df[TEXT_COL] = df[TEXT_COL].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


# ----------------------------
# 2) Auto-label helper (you can edit later)
# ----------------------------
def heuristic_label(text: str) -> int:
    """
    0 = mild overtime (e.g., 大小周 / occasional overtime)
    1 = standard 996/995
    2 = extreme exploitation (unpaid, forced, punishments, salary cuts, etc.)
    """
    t = text.lower()

    # Extreme signals (Class 2)
    extreme_keywords = [
        "无偿", "义务加班", "强制", "克扣", "降薪", "罚款", "惩罚",
        "不给", "不发", "拖欠", "违法", "严重违法", "996无偿",
        "绩效按加班", "加班不给钱", "加班不算", "没有加班费"
    ]
    if any(k in t for k in extreme_keywords):
        return 2

    # Standard 996-ish (Class 1)
    standard_keywords = ["996", "995", "007", "9-9-6", "9-9-7"]
    if any(k in t for k in standard_keywords):
        return 1

    # Mild signals (Class 0)
    mild_keywords = ["大小周", "加班", "调休", "偶尔加班", "周末加班"]
    if any(k in t for k in mild_keywords):
        return 0

    # Default to mild if unknown (you can relabel later)
    return 0


def export_labeling_file(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LABEL_COL not in df.columns:
        df["suggested_label"] = df[TEXT_COL].apply(heuristic_label)
        df[LABEL_COL] = ""  # blank for you to fill if you want
        df.to_csv(LABELING_CSV, index=False, encoding="utf-8-sig")
        print(f"[Saved] {LABELING_CSV}")
        print("Open it, fill the 'label' column with 0/1/2 (optional).")
        print("If you leave 'label' blank, training will use suggested_label.")
    return df


def load_labels_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """
    If you edited to_label_996.csv and filled LABEL_COL, we load it and use those labels.
    Otherwise we use suggested_label.
    """
    if os.path.exists(LABELING_CSV):
        edited = pd.read_csv(LABELING_CSV, encoding="utf-8-sig")
        # Ensure required columns exist
        if TEXT_COL in edited.columns:
            df = edited.copy()

    if "suggested_label" not in df.columns:
        df["suggested_label"] = df[TEXT_COL].apply(heuristic_label)

    # Build final y
    # If label is missing/blank -> use suggested_label
    def pick_label(row):
        val = str(row.get(LABEL_COL, "")).strip()
        if val == "" or val.lower() == "nan":
            return int(row["suggested_label"])
        return int(val)

    df["y"] = df.apply(pick_label, axis=1)
    return df


# ----------------------------
# 3) Build deep learning model (character-level)
# ----------------------------
def build_model(num_classes: int):
    vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        split="character",   # good default for Chinese
    )

    model = keras.Sequential([
        layers.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, vectorizer


# ----------------------------
# 4) Train + evaluate
# ----------------------------
def main():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"Can't find '{EXCEL_PATH}'. Put the Excel file next to this script."
        )

    df = load_and_clean(EXCEL_PATH)
    print("Loaded rows:", len(df))
    print("Columns:", list(df.columns))

    df = export_labeling_file(df)
    df = load_labels_if_present(df)

    texts = df[TEXT_COL].astype(str).values
    labels = df["y"].astype(int).values

    # Train/test split (stratify helps keep class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels if len(np.unique(labels)) > 1 else None
    )

    num_classes = len(np.unique(labels))
    print("Num classes:", num_classes, "Class counts:", dict(pd.Series(labels).value_counts()))

    model, vectorizer = build_model(num_classes=num_classes)

    # Adapt vectorizer on training data
    vectorizer.adapt(X_train)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    # Predictions
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model.save(MODEL_OUT, save_format="h5")
    print(f"\n[Saved model] {MODEL_OUT}")

    # Save training curve plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training Curve")
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=200)
        print("[Saved] training_curve.png")
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()