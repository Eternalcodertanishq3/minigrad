"""
06_mini_gpt.py — Character-level language model using miniGrad Transformers.

A ~50K parameter GPT-style model that learns to generate Shakespeare-like text.
Trains entirely on CPU using miniGrad's autograd engine.

Usage:
    python examples/06_mini_gpt.py

Architecture:
    Token Embedding + Positional Embedding
    → N × TransformerBlock (pre-norm, causal masked)
    → LayerNorm → Linear → logits

This proves miniGrad can train a real Transformer from scratch.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from minigrad.graph import no_grad
from minigrad.nn.attention import TransformerBlock
from minigrad.nn.embedding import Embedding
from minigrad.nn.layernorm import LayerNorm
from minigrad.nn.linear import Linear
from minigrad.nn.loss import CrossEntropyLoss
from minigrad.nn.module import Module
from minigrad.optim.adam import AdamW
from minigrad.tensor import Tensor

# ── Hyperparameters ──────────────────────────────────────────────────

EMBED_DIM = 64       # Model dimension
NUM_HEADS = 4        # Attention heads
NUM_LAYERS = 2       # Transformer blocks
BLOCK_SIZE = 32      # Context length (characters)
BATCH_SIZE = 16      # Training batch size
LEARNING_RATE = 3e-4
MAX_STEPS = 300      # Training steps
EVAL_INTERVAL = 50   # Print loss every N steps
GENERATE_LEN = 200   # Characters to generate

# ── Data ─────────────────────────────────────────────────────────────

# Tiny Shakespeare excerpt for demonstration
SHAKESPEARE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even to the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city is risen:
why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?
""".strip()


def build_dataset(text: str, block_size: int):
    """Build character-level dataset."""
    # Character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    # Encode the entire text
    data = np.array([stoi[ch] for ch in text], dtype=np.int64)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Dataset size:    {len(data)} characters")
    print(f"Sample chars:    {''.join(chars[:20])}...")

    return data, vocab_size, stoi, itos


def get_batch(data: np.ndarray, block_size: int, batch_size: int):
    """Sample a random batch of (input, target) pairs."""
    max_start = len(data) - block_size - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = np.stack([data[s:s + block_size] for s in starts])
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts])

    return Tensor(x.astype(np.float64)), Tensor(y.astype(np.float64))


# ── Model ────────────────────────────────────────────────────────────

class MiniGPT(Module):
    """
    A tiny GPT-style language model.

    Architecture:
        tok_emb + pos_emb → TransformerBlocks → LayerNorm → Linear head
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, block_size: int, dropout: float = 0.1):
        super().__init__()

        self.block_size = block_size

        # Embeddings
        self.tok_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(block_size, embed_dim)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ]

        # Final layer norm and projection to vocabulary
        self.ln_f = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            idx: Integer tensor of shape (B, T) with token indices

        Returns:
            Logits tensor of shape (B, T, vocab_size)
        """
        B, T = idx.data.shape[:2]

        # Token + positional embeddings
        tok = self.tok_emb(idx)                                    # (B, T, C)
        pos_indices = Tensor(np.arange(T, dtype=np.float64))       # (T,)
        pos = self.pos_emb(pos_indices)                            # (T, C)
        x = tok + pos                                              # (B, T, C) broadcast

        # Transformer blocks with causal masking
        for block in self.blocks:
            x = block(x, causal=True)

        # Final norm + project to vocab
        x = self.ln_f(x)
        logits = self.head(x)                                      # (B, T, vocab_size)

        return logits


def generate(model: MiniGPT, start_idx: np.ndarray, max_new_tokens: int,
             itos: dict) -> str:
    """
    Autoregressive text generation with greedy decoding.

    Args:
        model:          Trained MiniGPT model
        start_idx:      Starting token indices, shape (1, T)
        max_new_tokens: Number of new tokens to generate
        itos:           Index-to-character mapping

    Returns:
        Generated text string
    """
    model.eval()
    idx = start_idx.copy()

    with no_grad():
        for _ in range(max_new_tokens):
            # Crop context to block_size
            context = idx[:, -model.block_size:]
            logits = model(Tensor(context.astype(np.float64)))

            # Take logits at the last position
            last_logits = logits.data[0, -1, :]  # (vocab_size,)

            # Greedy: pick the highest probability token
            next_token = np.argmax(last_logits)
            idx = np.concatenate([idx, [[next_token]]], axis=1)

    # Decode
    return "".join(itos[int(t)] for t in idx[0])


# ── Training ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  miniGPT — Character-Level Language Model")
    print("  Powered by miniGrad's Transformer Engine")
    print("=" * 60)
    print()

    # Build dataset
    data, vocab_size, stoi, itos = build_dataset(SHAKESPEARE, BLOCK_SIZE)
    print()

    # Create model
    np.random.seed(42)
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=0.1,
    )

    # Count parameters
    num_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Architecture:     {NUM_LAYERS} layers, {NUM_HEADS} heads, {EMBED_DIM}d")
    print()

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = CrossEntropyLoss()

    # Training loop
    model.train()
    print("Training:")
    print("-" * 50)

    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        # Sample batch
        xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE)

        # Forward
        logits = model(xb)  # (B, T, vocab_size)

        # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
        B, T = xb.data.shape[:2]
        logits_flat = logits.reshape(B * T, vocab_size)
        targets_np = yb.data.reshape(B * T).astype(np.intp)
        loss = criterion(logits_flat, targets_np)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % EVAL_INTERVAL == 0 or step == 1:
            elapsed = time.time() - t0
            loss_val = float(loss.data)
            print(f"  step {step:>4d}/{MAX_STEPS} | loss: {loss_val:.4f} | "
                  f"time: {elapsed:.1f}s")

    total_time = time.time() - t0
    print("-" * 50)
    print(f"Training complete in {total_time:.1f}s")
    print()

    # ── Generation ───────────────────────────────────────────────────
    print("Generating text:")
    print("-" * 50)

    # Start with a newline (common starting character)
    start = np.array([[stoi["\n"]]], dtype=np.int64)
    generated = generate(model, start, GENERATE_LEN, itos)
    print(generated)
    print("-" * 50)
    print(f"Generated {len(generated)} characters")


if __name__ == "__main__":
    main()
