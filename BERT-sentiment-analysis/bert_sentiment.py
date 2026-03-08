# bert_sentiment.py
"""
IMDb Sentiment Analysis — BERT Fine-Tuning
===========================================
Binary sentiment classification on the IMDb movie reviews dataset
using a fine-tuned bert-base-uncased transformer model.

This project directly contrasts with the classical TF-IDF + Logistic
Regression approach in imdb_sentiment.py — same dataset, same task,
fundamentally different paradigm.

Pipeline:
    1. Load IMDb dataset (Hugging Face datasets)
    2. Tokenize with BERT WordPiece tokenizer
    3. Fine-tune bert-base-uncased with classification head (3 epochs)
    4. Evaluate (accuracy, F1, AUC-ROC)
    5. Attention visualization on sample reviews
    6. Direct comparison against classical baseline
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ------------------------------------------------
# Constants
# ------------------------------------------------

RANDOM_STATE    = 42
MODEL_NAME      = "bert-base-uncased"
MAX_LENGTH      = 256        # Max tokens per review (BERT max is 512, 256 balances speed/coverage)
BATCH_SIZE      = 16         # Safe for T4 GPU with MAX_LENGTH=256
EPOCHS          = 3          # Standard for BERT fine-tuning
LEARNING_RATE   = 2e-5       # Recommended range for BERT: 2e-5 to 5e-5
WARMUP_RATIO    = 0.1        # 10% of steps used for learning rate warmup
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# Data Loading and Tokenization
# ------------------------------------------------

def load_and_tokenize(
    model_name: str,
    max_length: int,
) -> tuple:
    """
    Load IMDb dataset and tokenize with BERT WordPiece tokenizer.

    BERT requires:
        - input_ids:      token IDs from WordPiece vocabulary
        - attention_mask: 1 for real tokens, 0 for padding
        - token_type_ids: segment IDs (not used for single-sequence tasks)

    Args:
        model_name: HuggingFace model identifier.
        max_length: Maximum sequence length (truncates/pads to this length).

    Returns:
        train_dataset, val_dataset, test_dataset, tokenizer
    """
    print(f"Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print("Tokenizing dataset (this may take a moment)...")
    tokenized = dataset.map(tokenize, batched=True, batch_size=512)
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    # Split training set into train/validation (90/10)
    train_val = tokenized["train"].train_test_split(
        test_size=0.1,
        seed=RANDOM_STATE,
    )

    print(f"Train samples:      {len(train_val['train']):,}")
    print(f"Validation samples: {len(train_val['test']):,}")
    print(f"Test samples:       {len(tokenized['test']):,}")

    return train_val["train"], train_val["test"], tokenized["test"], tokenizer


# ------------------------------------------------
# Model
# ------------------------------------------------

class BERTSentimentClassifier:
    """
    Fine-tuned BERT model for binary sentiment classification.

    BERT (Bidirectional Encoder Representations from Transformers) is
    pretrained on BookCorpus and English Wikipedia using masked language
    modeling and next sentence prediction. Fine-tuning adds a linear
    classification head on top of the [CLS] token representation.

    Key advantages over classical TF-IDF approach:
    - Contextual embeddings: "not good" is represented differently from "good"
    - Bidirectional context: each token attends to all other tokens
    - Transfer learning: leverages knowledge from 3.3B word pretraining corpus
    """

    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=True,
        )
        self.model.to(device)
        self.tokenizer = None
        print(f"Model loaded on: {device}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def fit(
        self,
        train_dataset,
        val_dataset,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        warmup_ratio: float = WARMUP_RATIO,
    ) -> None:
        """
        Fine-tune BERT with linear learning rate warmup and decay.

        Uses AdamW optimizer — a variant of Adam with corrected weight decay,
        standard for transformer fine-tuning.

        Linear warmup gradually increases the learning rate from 0 to the
        target LR over the first 10% of training steps, then linearly decays
        to 0. This stabilizes early training when weights are far from optimal.
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        total_steps   = len(train_loader) * epochs
        warmup_steps  = int(total_steps * warmup_ratio)
        scheduler     = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"\nFine-tuning BERT for {epochs} epochs")
        print(f"Total steps: {total_steps:,} | Warmup steps: {warmup_steps:,}")
        print("-" * 55)

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(train_loader):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if (step + 1) % 200 == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

            avg_train_loss = total_loss / len(train_loader)

            # --- Validation ---
            val_acc, val_f1 = self._quick_eval(val_loader)
            print(f"\nEpoch {epoch+1} complete | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n")

    def _quick_eval(self, loader: DataLoader) -> tuple[float, float]:
        """Fast accuracy and F1 evaluation on a DataLoader."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        return (
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds),
        )

    def evaluate(self, test_dataset, batch_size: int = BATCH_SIZE) -> None:
        """
        Full evaluation on test set: accuracy, F1, AUC-ROC,
        classification report, and confusion matrix.
        """
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs   = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                preds   = np.argmax(outputs.logits.cpu().numpy(), axis=1)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        print(f"\n{'='*55}")
        print("Test Results")
        print(f"{'='*55}")
        print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
        print(f"F1 Score:  {f1_score(all_labels, all_preds):.4f}")
        print(f"AUC-ROC:   {roc_auc_score(all_labels, all_probs):.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    def visualize_attention(
        self,
        text: str,
        tokenizer,
        save_path: str = "attention_visualization.png",
        layer: int = 11,
        head: int = 0,
    ) -> None:
        """
        Visualize BERT attention weights for a sample review.

        Attention weights show which tokens the model focuses on when
        making a prediction. The final layer (11) captures the most
        task-specific attention patterns after fine-tuning.

        Args:
            text:      Raw review string to visualize.
            tokenizer: BERT tokenizer used during training.
            save_path: Output path for the attention heatmap.
            layer:     Which transformer layer to visualize (0-11).
            head:      Which attention head to visualize (0-11).
        """
        self.model.eval()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract attention from specified layer and head
        attention = outputs.attentions[layer][0, head].cpu().numpy()
        tokens    = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        # Use [CLS] token attention row — what the classifier attended to
        cls_attention = attention[0]

        # Trim to first 30 tokens for readability
        n = min(30, len(tokens))
        tokens      = tokens[:n]
        cls_attention = cls_attention[:n]

        # Normalize
        cls_attention = cls_attention / cls_attention.max()

        pred_label = "Positive" if torch.argmax(outputs.logits).item() == 1 else "Negative"

        fig, ax = plt.subplots(figsize=(14, 3))
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=0, vmax=1)

        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            color = cmap(norm(score))
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax.text(i + 0.5, 0.5, token, ha="center", va="center",
                    fontsize=8, rotation=45, color="black")

        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(
            f"BERT Attention Weights — Predicted: {pred_label} | Layer {layer}, Head {head}",
            fontsize=12, pad=10
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.2, label="Attention weight")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Attention visualization saved to '{save_path}'")


# ------------------------------------------------
# Model Comparison
# ------------------------------------------------

def print_comparison() -> None:
    """
    Print a direct comparison between BERT and the classical baseline.
    Update the BERT results after running evaluate().
    """
    print("\n" + "="*55)
    print("Model Comparison — IMDb Sentiment")
    print("="*55)
    print(f"{'Model':<35} {'Accuracy':>9} {'F1':>7} {'AUC':>7}")
    print("-"*55)
    print(f"{'TF-IDF + Logistic Regression':<35} {'~0.89':>9} {'~0.89':>7} {'~0.96':>7}")
    print(f"{'BERT (bert-base-uncased)':<35} {'—':>9} {'—':>7} {'—':>7}")
    print("-"*55)
    print("Fill in BERT results after running evaluate()")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    print(f"Device: {DEVICE}")

    # 1. Load and tokenize
    train_dataset, val_dataset, test_dataset, tokenizer = load_and_tokenize(
        MODEL_NAME, MAX_LENGTH
    )

    # 2. Initialize model
    classifier = BERTSentimentClassifier(MODEL_NAME, DEVICE)
    classifier.tokenizer = tokenizer

    # 3. Fine-tune
    classifier.fit(
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )

    # 4. Evaluate on held-out test set
    classifier.evaluate(test_dataset)

    # 5. Attention visualization on a sample review
    sample_positive = "This film was absolutely brilliant. The performances were outstanding and the story kept me engaged throughout."
    sample_negative = "What a waste of time. The plot made no sense and the acting was terrible from start to finish."

    classifier.visualize_attention(sample_positive, tokenizer, "attention_positive.png")
    classifier.visualize_attention(sample_negative, tokenizer, "attention_negative.png")

    # 6. Print model comparison
    print_comparison()
