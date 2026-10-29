"""Character- or word-level next-token prediction task for the arm eval.

REAL CORPUS — this module depends on a file OUTSIDE the repo, by design:

    /Users/id19/Programming/ml_ai_and_data_science/Karpathy Zero to Hero/Shakespeare.txt

(~1.1 MB of public-domain Shakespeare). The old embedded retelling is gone:
make_data raises if the corpus file is missing (pass corpus_path=... to
point at a copy elsewhere, e.g. on a different machine).

Pipeline kept from the toy version: build vocab, cut non-overlapping
windows of 128 tokens, split the TOKEN STREAM chronologically 80/20 (so no
window crosses the train/val boundary), y = the same ids; the runner
(run_eval.py) does the one-position shift inside the loss.

One materialisation change forced by the real corpus: the word-level vocab
(~25k) as dense (N, 128, V) one-hot floats would need tens of GB, so
make_data returns x as long token-id windows (N, 128) and the runner
one-hots THEM PER BATCH. At the model boundary the input is still exactly
the documented one-hot (B, L, d_in) — only where the one-hot is built
moved. The per-batch one-hot is the CANONICAL exact float one-hot
(run_eval.one_hot_batch): bit-identical to F.one_hot(ids, V).float(), and
at a Linear boundary exactly equal to an embedding lookup, ids -> W(ids) + b.
"""

from pathlib import Path

import torch

NAME = "text"
SEQ_LEN = 128

CORPUS_PATH = Path(
    "/Users/id19/Programming/ml_ai_and_data_science/Karpathy Zero to Hero/Shakespeare.txt"
)


def tokenize(text, level="char"):
    if level == "word":
        return text.lower().split()
    return list(text)  # char level: one token per character


def make_data(level="char", seq_len=SEQ_LEN, seed=0, corpus_path=CORPUS_PATH):
    corpus_path = Path(corpus_path)
    if not corpus_path.is_file():
        raise ValueError(
            f"text corpus not found: {corpus_path}\n"
            "expected the real Shakespeare corpus (see this module's docstring); "
            "pass corpus_path=... with another copy of the file"
        )
    text = corpus_path.read_text(encoding="utf-8")

    tokens = tokenize(text, level)

    # split the TOKEN STREAM chronologically first, then window each side,
    # so no window ever crosses the train/val boundary
    n_train = int(len(tokens) * 0.8)
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:]

    vocab = sorted(set(tokens))
    tok2id = {t: i for i, t in enumerate(vocab)}
    id2tok = {i: t for i, t in enumerate(vocab)}  # kept for inspection

    def windows(tok_list):
        # non-overlapping windows of seq_len tokens, kept as long ids —
        # the runner expands them to one-hot per batch (see docstring)
        xs, ys = [], []
        for start in range(0, len(tok_list) - seq_len + 1, seq_len):
            chunk = [tok2id[t] for t in tok_list[start:start + seq_len]]
            xs.append(chunk)
            ys.append(chunk)  # same tokens; runner shifts by one
        x = torch.tensor(xs, dtype=torch.long)   # (N, seq_len) token ids
        y = torch.tensor(ys, dtype=torch.long)   # (N, seq_len)
        return x, y

    train_x, train_y = windows(train_tokens)
    val_x, val_y = windows(val_tokens)

    return {
        "train": train_x,
        "train_y": train_y,
        "val": val_x,
        "val_y": val_y,
        "d_in": len(vocab),
        "d_out": len(vocab),  # logits width for "ce"
        "loss": "ce",
        "level": level,
        "id2tok": id2tok,
        "corpus_path": str(corpus_path),
    }


if __name__ == "__main__":
    d = make_data()
    name = f"{NAME}_{d['level']}"
    print(
        f"{name}: train {tuple(d['train'].shape)} y {tuple(d['train_y'].shape)} "
        f"val {tuple(d['val'].shape)} d_in={d['d_in']} d_out={d['d_out']} loss={d['loss']}"
    )