# bert_sentiment_with_interview_notes.py
"""
NOTES TO INTERVIEWER:

This project fine-tunes bert-base-uncased on the IMDb movie reviews dataset
for binary sentiment classification.

It is designed as a direct contrast to the classical TF-IDF + Logistic
Regression approach in imdb_sentiment.py — same dataset, same task,
fundamentally different paradigm.

The key story this project tells:
- Classical NLP (TF-IDF) treats words as independent frequency counts
- BERT understands words in context, bidirectionally, with 110M parameters
  of pretrained world knowledge
- The accuracy gap (~4%) demonstrates what transfer learning adds over
  a strong classical baseline
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
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

"""
NOTES TO INTERVIEWER:

AdamW is imported from torch.optim rather than transformers. In newer
versions of the transformers library (4.x+), AdamW was moved to PyTorch
core — importing it from transformers raises an ImportError.

The transformers library provides:
- BertTokenizer: WordPiece tokenizer pretrained with bert-base-uncased
- BertForSequenceClassification: BERT with a linear classification head
  on top of the [CLS] token
- get_linear_schedule_with_warmup: learning rate scheduler standard for
  transformer fine-tuning

datasets is Hugging Face's library for loading benchmark datasets with
one line of code. It handles downloading, caching, and efficient batched
processing automatically.
"""


# ------------------------------------------------
# Constants
# ------------------------------------------------

RANDOM_STATE    = 42
MODEL_NAME      = "bert-base-uncased"
MAX_LENGTH      = 256
BATCH_SIZE      = 16
EPOCHS          = 3
LEARNING_RATE   = 2e-5
WARMUP_RATIO    = 0.1
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
NOTES TO INTERVIEWER:

MAX_LENGTH = 256 is a deliberate tradeoff. BERT supports up to 512 tokens,
but doubling the sequence length roughly doubles memory usage and training
time with minimal accuracy gain on sentiment classification — most of the
sentiment signal is in the first half of a review anyway.

BATCH_SIZE = 16 is the safe upper limit for a T4 GPU at MAX_LENGTH=256.
Larger batches would cause out-of-memory errors on free Colab.

LEARNING_RATE = 2e-5 is at the lower end of the recommended range for
BERT fine-tuning (2e-5 to 5e-5). Lower learning rates mean smaller weight
updates — important because BERT's weights are already well-calibrated from
pretraining and aggressive updates would destroy that prior knowledge.
This is sometimes called "catastrophic forgetting" — fine-tuning too fast
causes the model to lose its pretrained representations.

WARMUP_RATIO = 0.1 means the first 10% of training steps gradually increase
the learning rate from 0 to the target LR. This stabilizes early training
before the model has had enough gradient signal to know which direction to
update weights confidently.

DEVICE automatically uses GPU if available, CPU otherwise. In Colab, this
will detect the T4 GPU. On a local machine without a GPU this will fall
back to CPU, which is much slower but functionally correct.
"""


# ------------------------------------------------
# Data Loading and Tokenization
# ------------------------------------------------

def load_and_tokenize(model_name: str, max_length: int) -> tuple:
    """
    Load IMDb dataset and tokenize with BERT WordPiece tokenizer.

    BERT requires three inputs per sequence:
        - input_ids:      integer token IDs from the WordPiece vocabulary
        - attention_mask: 1 for real tokens, 0 for padding tokens
        - token_type_ids: segment IDs for sentence pairs (not used here)

    Args:
        model_name: HuggingFace model identifier.
        max_length: Maximum sequence length in tokens.

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

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize, batched=True, batch_size=512)
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    train_val = tokenized["train"].train_test_split(test_size=0.1, seed=RANDOM_STATE)

    print(f"Train:      {len(train_val['train']):,}")
    print(f"Validation: {len(train_val['test']):,}")
    print(f"Test:       {len(tokenized['test']):,}")

    return train_val["train"], train_val["test"], tokenized["test"], tokenizer

"""
NOTES TO INTERVIEWER:

WordPiece tokenization is fundamentally different from the TF-IDF approach
in the classical project. TF-IDF treats each word as an atomic unit —
"running" and "run" are completely different features. WordPiece breaks
words into subword units: "running" → ["run", "##ning"]. This means:

1. Out-of-vocabulary words are handled gracefully — any word can be
   represented as a sequence of known subword pieces
2. Morphological relationships are captured — "run", "running", "runner"
   share the "run" subword piece
3. The vocabulary is fixed at 30,522 tokens and covers virtually any
   English text

The [CLS] token is prepended to every sequence and [SEP] appended.
After fine-tuning, the [CLS] representation captures the overall
sentiment of the review and is fed into the classification head.

padding="max_length" pads all sequences to exactly MAX_LENGTH tokens.
truncation=True cuts sequences longer than MAX_LENGTH. The attention_mask
tells BERT which positions are real tokens (1) vs padding (0), so it
doesn't attend to padding positions.

dataset.map() with batched=True processes 512 reviews at a time, which
is much faster than tokenizing one review at a time.
"""


# ------------------------------------------------
# Model
# ------------------------------------------------

class BERTSentimentClassifier:
    """
    Fine-tuned BERT model for binary sentiment classification.

    BERT (Bidirectional Encoder Representations from Transformers):
    - Pretrained on BookCorpus + English Wikipedia (~3.3B words)
    - 12 transformer layers, 12 attention heads, 768 hidden dimensions
    - 110 million trainable parameters
    - Fine-tuning adds a linear layer: 768 → 2 (positive/negative)

    The key advantage over classical models: BERT produces contextual
    embeddings — the representation of "not" in "not good" is different
    from "not" in "not bad" because it attends to surrounding tokens.
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
        print(f"Device: {device}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    """
    NOTES TO INTERVIEWER:

    BertForSequenceClassification loads the pretrained BERT weights and
    adds a randomly initialized linear classification head:
        Linear(768, 2) → outputs logits for [negative, positive]

    num_labels=2 sets the output size of this head.

    output_attentions=True tells the model to return attention weight
    tensors at every layer during the forward pass. This is needed for
    the attention visualization later — it adds a small memory overhead
    but is essential for interpretability.

    .to(device) moves all model parameters to GPU memory. This is a
    single call that moves all 110M parameters at once.

    The parameter count print is useful for interviews — bert-base-uncased
    has ~110M parameters compared to the ~50k parameters in the breast
    cancer neural network. The scale difference illustrates why pretraining
    is necessary — you cannot train 110M parameters from scratch on 25k
    reviews without catastrophic overfitting.
    """

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
        Fine-tune BERT with AdamW optimizer and linear warmup scheduler.

        Training loop:
            1. Zero gradients
            2. Forward pass (compute loss internally via labels argument)
            3. Backward pass (compute gradients)
            4. Clip gradients to max norm 1.0
            5. Update weights
            6. Step learning rate scheduler
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        total_steps  = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"\nFine-tuning BERT for {epochs} epochs")
        print(f"Total steps: {total_steps:,} | Warmup steps: {warmup_steps:,}")
        print("-" * 55)

        for epoch in range(epochs):
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
            val_acc, val_f1 = self._quick_eval(val_loader)
            print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n")

    """
    NOTES TO INTERVIEWER:

    AdamW differs from standard Adam in how it applies weight decay.
    Standard Adam folds weight decay into the gradient update, which
    interacts incorrectly with adaptive learning rates. AdamW applies
    weight decay directly to the weights after the gradient step,
    which is mathematically correct and empirically stronger for
    transformer fine-tuning.

    Passing labels= directly to the model is a Hugging Face convenience —
    BertForSequenceClassification computes cross-entropy loss internally
    and returns it in outputs.loss. This is equivalent to computing
    CrossEntropyLoss(outputs.logits, labels) manually.

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) rescales
    all gradients so their total norm doesn't exceed 1.0. Without this,
    occasional large gradient steps can destabilize training — particularly
    important with 12 layers of backpropagation where gradients can compound.

    scheduler.step() must be called after optimizer.step() — this is a
    common mistake. The scheduler adjusts the learning rate based on the
    current step count, linearly decaying from the peak LR back toward 0.

    shuffle=True on the training DataLoader randomizes the order of
    reviews each epoch. Without shuffling, the model would see all positive
    reviews before all negative ones in each epoch, causing unstable gradients.
    """

    def _quick_eval(self, loader: DataLoader) -> tuple[float, float]:
        """Fast accuracy and F1 on a DataLoader without storing probabilities."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"]
                outputs        = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds          = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds)

    """
    NOTES TO INTERVIEWER:

    torch.no_grad() disables gradient computation during evaluation.
    This is essential — without it, PyTorch builds a computation graph
    for every forward pass, consuming GPU memory unnecessarily. During
    inference we only need the output values, not the gradients.

    self.model.eval() switches the model to evaluation mode. This affects
    two layer types:
    1. Dropout layers: disabled during eval (all neurons active)
    2. BatchNorm layers: uses running statistics rather than batch statistics
    BERT uses dropout but not BatchNorm, so eval() primarily affects dropout.

    torch.argmax(outputs.logits, dim=1) picks the class with the higher
    logit — equivalent to argmax(softmax(logits)) but more numerically
    efficient since softmax doesn't change the argmax.
    """

    def evaluate(self, test_dataset, batch_size: int = BATCH_SIZE) -> None:
        """Full test set evaluation with F1, AUC-ROC, and confusion matrix."""
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"]
                outputs        = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs          = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                preds          = np.argmax(outputs.logits.cpu().numpy(), axis=1)
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

    """
    NOTES TO INTERVIEWER:

    torch.softmax(outputs.logits, dim=1)[:, 1] converts raw logits to
    probabilities and extracts the positive class probability. This is
    needed for AUC-ROC, which requires continuous probability scores
    rather than hard class predictions.

    I report three metrics:
    1. Accuracy — interpretable but assumes balanced classes (IMDb is 50/50
       so accuracy is meaningful here, unlike the hospital readmission project)
    2. F1 Score — harmonic mean of precision and recall on the positive class
    3. AUC-ROC — measures ranking quality across all thresholds; best for
       comparing models regardless of decision threshold

    The comparison with the classical baseline is the key takeaway:
    BERT achieves ~93% accuracy vs ~89% for TF-IDF + Logistic Regression.
    The 4% gap represents what 110M parameters of pretrained language
    understanding adds over a bag-of-words frequency count approach.
    """

    def visualize_attention(
        self,
        text: str,
        tokenizer,
        save_path: str = "attention_visualization.png",
        layer: int = 11,
        head: int = 0,
    ) -> None:
        """
        Visualize [CLS] token attention weights for a sample review.

        The [CLS] token attends to all other tokens to build its
        summary representation. Visualizing this shows which words
        most influenced the sentiment prediction.
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

        attention   = outputs.attentions[layer][0, head].cpu().numpy()
        tokens      = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        cls_attention = attention[0]

        n             = min(30, len(tokens))
        tokens        = tokens[:n]
        cls_attention = cls_attention[:n]
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
            f"BERT Attention — Predicted: {pred_label} | Layer {layer}, Head {head}",
            fontsize=12, pad=10,
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.2, label="Attention weight")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Attention visualization saved to '{save_path}'")

    """
    NOTES TO INTERVIEWER:

    BERT has 12 transformer layers, each with 12 attention heads — 144
    attention patterns in total. Each head learns to attend to different
    linguistic relationships. Some heads track syntactic dependencies,
    others track coreference, others track semantic similarity.

    I visualize layer 11 (the final layer) because by this point the
    attention has been shaped by fine-tuning to focus on sentiment-relevant
    tokens. Earlier layers tend to capture more general linguistic patterns.

    The [CLS] token's attention row (attention[0]) is the most meaningful
    for classification because [CLS] is the token whose final representation
    is fed into the classification head. Its attention weights show what
    information it gathered from the rest of the sequence.

    outputs.attentions is a tuple of 12 tensors, one per layer.
    Each tensor has shape: (batch_size, num_heads, seq_len, seq_len)
    So outputs.attentions[11][0, 0] gives layer 12, batch item 0, head 0.

    The colormap (RdYlGn — red to green) makes high-attention tokens
    immediately visible as green and low-attention tokens as red. This
    is the kind of visualization that works well in interviews and
    presentations because it makes an abstract mechanism concrete.
    """


# ------------------------------------------------
# Comparison
# ------------------------------------------------

def print_comparison() -> None:
    """Print model comparison table. Fill in BERT results after evaluate()."""
    print("\n" + "="*55)
    print("Model Comparison — IMDb Sentiment")
    print("="*55)
    print(f"{'Model':<35} {'Accuracy':>9} {'F1':>7} {'AUC':>7}")
    print("-"*55)
    print(f"{'TF-IDF + Logistic Regression':<35} {'~0.89':>9} {'~0.89':>7} {'~0.96':>7}")
    print(f"{'BERT (bert-base-uncased)':<35} {'—':>9} {'—':>7} {'—':>7}")
    print("-"*55)
    print("Fill in BERT results after running evaluate()")

"""
NOTES TO INTERVIEWER:

The comparison table tells the core story of this project:
- TF-IDF is a strong classical baseline (~89% accuracy is genuinely good)
- BERT adds ~4% accuracy by understanding context and using pretrained knowledge
- The accuracy gain comes at a significant cost: training time (~30 min on GPU
  vs ~10 seconds for logistic regression) and inference cost (~50ms per review
  vs <1ms)

This tradeoff is real and worth discussing in interviews. For a production
sentiment classifier handling millions of reviews per day, TF-IDF + Logistic
Regression might be the better engineering choice. BERT is worth the cost
when accuracy is critical, data is limited, or the text is complex enough
that context matters substantially (sarcasm, negation, domain-specific language).

Smaller, distilled transformer models (DistilBERT, TinyBERT) can close
this gap — DistilBERT achieves ~97% of BERT's performance at 40% of the
model size and 60% faster inference.
"""


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    """
    NOTES TO INTERVIEWER:

    The pipeline is deliberately structured to mirror the classical project
    for direct comparison:
    1. Load and tokenize — equivalent to vectorize in the classical project
    2. Initialize model — equivalent to instantiating LogisticRegression
    3. Fine-tune — equivalent to .fit()
    4. Evaluate — equivalent to .evaluate()
    5. Visualize — attention plots replace feature coefficient plots

    The test set is never touched until step 4, consistent with the
    data leakage prevention approach across all projects in this portfolio.
    """

    print(f"Device: {DEVICE}")

    # 1. Load and tokenize
    train_dataset, val_dataset, test_dataset, tokenizer = load_and_tokenize(
        MODEL_NAME, MAX_LENGTH
    )

    # 2. Initialize model
    classifier = BERTSentimentClassifier(MODEL_NAME, DEVICE)
    classifier.tokenizer = tokenizer

    # 3. Fine-tune
    classifier.fit(train_dataset, val_dataset)

    # 4. Evaluate
    classifier.evaluate(test_dataset)

    # 5. Attention visualization
    sample_positive = "This film was absolutely brilliant. The performances were outstanding and the story kept me engaged throughout."
    sample_negative = "What a waste of time. The plot made no sense and the acting was terrible from start to finish."

    classifier.visualize_attention(sample_positive, tokenizer, "attention_positive.png")
    classifier.visualize_attention(sample_negative, tokenizer, "attention_negative.png")

    # 6. Comparison
    print_comparison()
