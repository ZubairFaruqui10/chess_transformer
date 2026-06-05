# Chess Move Generation with Transformers

A Transformer-based model that learns to predict strong chess moves given a board state — formally learning the conditional distribution **p(move | board)** — trained purely from (board, move) pairs with no tree search or lookahead. 

---

## Motivation

Chess is a fully observable, deterministic game with objective move quality, making it an ideal testbed for conditional generative modeling. Move quality is measurable via [Stockfish](https://stockfishchess.org) in centipawns (cp), unlike subjective text/image generation. The same (state, action) conditional modeling framework extends naturally to robotics, navigation, and other sequential decision-making domains.

---

## Dataset

- **Source:** [Lichess Open Game Database](https://database.lichess.org) — January 2013 file (~120K games)
- **Filter:** 50,000 games with Elo ≥ 1,500 for both players
- **Train/Val split:** 90/10

**Board Encoding:** Each position → length-64 integer vector using a 13-symbol vocabulary:

| Token | Meaning |
|---|---|
| 0 | Empty square |
| 1–6 | White P, N, B, R, Q, K |
| 7–12 | Black p, n, b, r, q, k |

**Move Encoding:** `from_square × 64 + to_square` → vocabulary of **4,096** possible moves

---

## Model Architecture

**ChessTransformer** — Transformer encoder treating the board as a sequence of 64 tokens.

| Component | Details |
|---|---|
| Embedding | 256-dim learned embeddings per square token |
| Positional Encoding | Sine/cosine over 64 positions |
| Transformer Encoder | 6 layers, 8-head self-attention, FF dim 1024, dropout 0.1 |
| Aggregation | Mean-pool over 64 output tokens |
| Classification Head | Linear → 4,096 logits (one per move) |
| Parameters | ~5.8 M |
| Framework | PyTorch, CUDA |

Self-attention over all 64 squares captures long-range piece interactions without the locality bias of CNNs.

---

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| LR Schedule | Cosine annealing |
| Loss | Cross-entropy (4,096 classes) |
| Batch Size | 512 |
| Epochs | 10 |
| Gradient Clipping | 1.0 |

Best checkpoint saved based on validation loss (`chess_transformer_best.pt`).

---

## Results

| Metric | Value |
|---|---|
| Legal Move Rate | 99.83% |
| Top-5 Stockfish Match Rate | 67.9% |
| Avg Centipawn Loss | 221.7 cp |

**Reference centipawn losses:** Grandmaster ~20 cp · Club player ~50–80 cp · Random legal moves ~300+ cp

---

## Self-Play Demo

The model plays both sides from the starting position with legal-move masking. The resulting game shows recognizable opening theory (Ruy Lopez) and a clean tactical finish:

```
1. e4 e5  2. Nf3 Nc6  3. Bb5 a6  4. Ba4 Nf6  5. O-O Be7
6. Re1 b5  7. Bb3 O-O  8. c3 d5   9. exd5 e4  10. Ng5 Na5
11. Bc2 Bg4  12. f3 exf3  13. gxf3 Bh5  14. d4 Nxd5  15. f4 g6
16. Qd3 Bd6  17. f5 gxf5  18. Qxf5 Kh8  19. Qf6+ Kg8  20. Bxh7#
```

Result: **1–0 (White wins)**. Stockfish confirms the final position is a forced checkmate.

---

## Limitations

- **No game context:** model sees only the current board — cannot detect repetition or plan multi-move combinations
- **Promotion encoding:** promotion piece is ignored (e.g. `e7e8q` treated as `e7e8`)
- **Castling / en-passant:** require auxiliary board state not present in the 64-token encoding
- **Dataset scale:** January 2013 is a small slice; a modern full-month file (~100M positions) would substantially improve results

---

## Project Structure

```
genAIProject/
├── chess_transformer_final.ipynb          # Full pipeline: data → train → eval → demo
├── chess_transformer.ipynb                # Development notebook
├── chess_transformer_best.pt              # Best model checkpoint
├── lichess_db_standard_rated_2013-01.pgn  # Training data (PGN)
├── training_curves.png                    # Loss and accuracy curves
└── evaluation_results.png                 # Centipawn loss distribution + metrics
```

---

## Tech Stack

- Python, PyTorch
- [python-chess](https://python-chess.readthedocs.io) — board parsing, legal move generation
- [Stockfish](https://stockfishchess.org) — move quality evaluation
- [Lichess Open Game Database](https://database.lichess.org) — training data

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Lichess Open Game Database — https://database.lichess.org
- python-chess — https://python-chess.readthedocs.io
