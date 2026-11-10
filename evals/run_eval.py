"""Shared toy eval runner (tasks + train loop). Arms come from models_*.py.

Usage:
    .venv/bin/python evals/run_eval.py --family deltanet --task sine --mock
    .venv/bin/python evals/run_eval.py --family deltanet --task sine --arm la
    .venv/bin/python evals/run_eval.py --family attention --task sine --arm vanilla

--family is deltanet or attention. --task is a comma list or 'all'
(sine, text_char, text_word, audio).

--mock runs a dummy model to prove the plumbing. With --mock, --steps
defaults to 200 and a single 'mock' arm replaces the arm list.

BATCHING: every pass, train and val, is batched. Training uses --batch
(16); the final no-grad val pass uses the smaller --val-batch (8) so peak
memory stays flat even at the word-level vocab (~23.6k). Text tasks pass
(B, L) int64 token ids to the model; the model owns an nn.Embedding
(width 64, identical shape across arms) — mathematically identical to
the one-hot @ embedding-matrix form (was verified exactly), just the
honest cheap shape.
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

import task_audio
import task_sine
import task_text

TASKS = ["sine", "text_char", "text_word", "audio"]
ARMS = None  # from models_deltanet / models_attention in main()
ArmsModel = None


class MockModel(torch.nn.Module):
    """DUMMY RANDOM MODEL used by --mock to prove the plumbing end to end.

    It is ALSO the living spec of the EXACT interface the real models.py
    must honour (build_model below is the single adapter; when models.py
    lands, only that function changes):

      - forward(x) takes raw (B, L) int64 token ids for the text tasks
        (loss "ce") and embeds them ITSELF: the model owns an
        nn.Embedding(d_in, 64) — width 64, identical shape across arms —
        so the model-visible width is 64 and the output stays full-vocab
        logits (B, L, d_out). Mathematically identical to the old one-hot
        @ embedding-matrix form (was verified exactly), just the honest
        cheap shape. For sine and audio (loss "mse") x is raw (B, L, d_in)
        float frames/samples and the output is (B, L, d_out) per-token
        predictions;
      - it is a torch.nn.Module, so .parameters() yields the trainable
        tensors (counted for the params column and handed to AdamW);
      - it is constructed as Model(d_in, d_out, loss).

    The ce variant is emb(ids) -> (Linear 64->64, SiLU, Linear 64->d_out),
    so the losses clearly move rather than sitting flat; the mse variant
    is a plain random Linear projection, which also trains on the
    next-frame tasks.
    """

    def __init__(self, d_in, d_out, loss):
        super().__init__()
        self.loss = loss
        if loss == "ce":
            self.emb = torch.nn.Embedding(d_in, 64)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, d_out),
            )
        else:
            self.net = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        if self.loss == "ce":
            x = self.emb(x)
        return self.net(x)


def build_model(args, arm, d_in, d_out, loss):
    if args.mock:
        return MockModel(d_in, d_out, loss)  # arm is meaningless for the mock
    return ArmsModel(arm, d_in, d_out, loss)


def batch_loss(model, loss, x, y):
    # shift lives here, in the runner: the factories return raw series/tokens.
    # CE alignment: input position i enters the model and predicts target
    # token i+1 -> logits[:, :-1] against y[:, 1:]. mse: frame i predicts
    # frame i+1 -> model(x[:, :-1]) against x[:, 1:]. Same shift for both.
    if loss == "ce":
        # token-id windows (B, L) go straight to the model: the MODEL owns
        # the nn.Embedding (width 64, identical shape across arms); the
        # runner never builds (B, L, V) inputs.
        logits = model(x)  # (B, L, V): position i predicts token i+1
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1)
        )
    pred = model(x[:, :-1])            # (B, L-1, d_out) = next-sample prediction
    return ((pred - x[:, 1:]) ** 2).mean()


def train_arm(model, loss, x_train, y_train, x_val, y_val, args):
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(args.seed)
    n = x_train.shape[0]

    recent = []
    print_every = max(1, args.steps // 10)
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), generator=g)
        xb = x_train[idx]
        yb = y_train[idx] if y_train is not None else None
        step_loss = batch_loss(model, loss, xb, yb)
        opt.zero_grad()
        step_loss.backward()
        opt.step()
        recent.append(step_loss.item())
        if len(recent) > 100:
            recent.pop(0)
        if step % print_every == 0:
            print(f"  step {step:5d}  train(mean100) {sum(recent) / len(recent):.4f}")

    train_loss = sum(recent) / len(recent)  # running mean over the last 100 steps

    # final val pass, no grad, BATCHED like training (never a full-set
    # forward: at word level the logits for the whole val set would be
    # (N, 128, 23641) in one go). val_batch is deliberately smaller than
    # the train batch to keep peak memory flat at both vocab levels.
    with torch.no_grad():
        total, count = 0.0, 0
        for start in range(0, x_val.shape[0], args.val_batch):
            xb = x_val[start:start + args.val_batch]
            yb = y_val[start:start + args.val_batch] if y_val is not None else None
            total += batch_loss(model, loss, xb, yb).item() * xb.shape[0]
            count += xb.shape[0]
    # position-weighted mean over windows: every window contributes the
    # same number of predicted positions (L-1), so weighting by windows
    # keeps the final partial batch (N % val_batch) from being over-counted
    val_loss = total / count
    return train_loss, val_loss


def get_data(args, task):
    if task == "sine":
        return task_sine.make_data(seed=args.seed)
    if task == "text_char":
        return task_text.make_data(level="char", seed=args.seed)
    if task == "text_word":
        return task_text.make_data(level="word", seed=args.seed)
    if task == "audio":
        return task_audio.make_data(audio_path=args.audio, seed=args.seed)
    raise ValueError(f"unknown task: {task}")


def run_task(args, task):
    t_data = time.perf_counter()
    d = get_data(args, task)
    t_data = time.perf_counter() - t_data
    loss = d["loss"]
    y_train = d.get("train_y")
    y_val = d.get("val_y")

    print("=" * 64)
    print(f"family={args.family} task={task} loss={loss} d_in={d['d_in']} d_out={d['d_out']} "
          f"train={tuple(d['train'].shape)} val={tuple(d['val'].shape)}")
    if "source" in d:  # audio only: which file actually got used
        extra = f" ({d['flac_seconds']} s)" if d.get("flac_seconds") else ""
        print(f"audio source: {d['source']}{extra}")
    print(f"data build time: {t_data:.2f} s")
    print("=" * 64)

    arms = ["mock"] if args.mock else (ARMS if args.arm == "all" else [args.arm])
    results = []
    for arm in arms:
        print(f"== {arm} ==")
        torch.manual_seed(args.seed)  # every arm starts identically
        model = build_model(args, arm, d["d_in"], d["d_out"], loss)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t_arm = time.perf_counter()
        train_loss, val_loss = train_arm(
            model, loss, d["train"], y_train, d["val"], y_val, args,
        )
        t_arm = time.perf_counter() - t_arm
        print(f"  done: train {train_loss:.4f}  val {val_loss:.4f}")
        results.append((arm, n_params, train_loss, val_loss, t_arm))

    # summary table for THIS task
    print()
    header = f"{'arm':<10} {'params':>10} {'train_loss':>12} {'val_loss':>12} {'time_s':>8}"
    print(header)
    print("-" * len(header))
    for arm, n_params, train_loss, val_loss, t_arm in results:
        print(f"{arm:<10} {n_params:>10} {train_loss:>12.4f} {val_loss:>12.4f} {t_arm:>8.2f}")
    print()


def parse_tasks(arg):
    if arg.strip().lower() == "all":
        return list(TASKS)
    tasks = [t.strip() for t in arg.split(",") if t.strip()]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        raise ValueError(
            f"unknown task(s) {bad}; valid: {TASKS} (or 'all')"
        )
    return tasks


def main():
    ap = argparse.ArgumentParser(description="Compare model-family arms on shared toy tasks")
    ap.add_argument("--family", default="deltanet", choices=["deltanet", "attention"],
                    help="which arms to train (deltanet or attention)")
    ap.add_argument("--task", default="all",
                    help=f"comma list or 'all' (= {','.join(TASKS)})")
    ap.add_argument("--arm", default="all",
                    help="one arm name, or 'all' (names depend on --family)")
    ap.add_argument("--audio", default=None,
                    help="path to an audio file; only used with task audio")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (default: 200 with --mock, else 3000)")
    ap.add_argument("--mock", action="store_true",
                    help="use MockModel instead of the real arms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-batch", type=int, default=8,
                    help="batch size for the (no-grad) val pass; smaller "
                         "than --batch so the word-level val peak memory "
                         "(logits span the ~23.6k vocab) stays low")
    args = ap.parse_args()

    args.steps = (200 if args.mock else 3000) if args.steps is None else args.steps

    try:
        tasks = parse_tasks(args.task)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    if args.audio and "audio" not in tasks:
        print("--audio is only used with task audio")
        sys.exit(1)

    if not args.mock:
        global ARMS, ArmsModel
        if args.family == "attention":
            from models_attention import ArmsModel, ARMS
        else:
            from models_deltanet import ArmsModel, ARMS
        if args.arm != "all" and args.arm not in ARMS:
            print(f"unknown arm {args.arm!r} for family {args.family}; valid: {ARMS} (or 'all')")
            sys.exit(1)

    t_suite = time.perf_counter()
    for task in tasks:
        run_task(args, task)
    print(f"total suite time: {time.perf_counter() - t_suite:.2f} s")


if __name__ == "__main__":
    main()