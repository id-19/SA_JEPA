# SA-JEPA v2 — BUILD LOG (plain-language, from basic parts)

Written 2026-10-15. Audience: someone comfortable with basic ML components (a conv, an
attention layer, a loss function) but new to JEPA, delta-net state, or SIGREG. Everything
below describes the code in this folder as it actually is — no aspiration.

---

## 0. What we are building, in one paragraph

A model that watches a stream of audio and learns, with **no labels at all**, to predict what
its own internal summary of the audio will look like a moment later. Training it produces an
**encoder**: a function that turns any audio clip into a sequence of 256-number vectors
("latents"), one vector per 20 ms. If those vectors are good, they can be used for other
things — identifying who is speaking, telling speech from music, and (with a small decoder)
reconstructing the audio. That last part is what makes the result demonstrable rather than
merely numeric.

The design has three parts: an **encoder** (audio → latents), a **predictor** (latents →
future latents), and a **regularizer** (SIGREG) whose only job is to stop the first two from
cheating.

---

## 1. Input: the mel spectrogram

Raw audio at 16 kHz is 16,000 numbers per second — too many, and mostly redundant. We
convert it to a **mel spectrogram** in three steps:

1. **STFT**: slide a 25 ms window (400 samples) every 20 ms (hop = 320 samples), and for each
   position compute which frequencies are present. Output: 201 frequency bins per frame.
2. **Mel filterbank**: squash those 201 bins into 80 bins, using filters spaced the way human
   hearing is (finer at low frequencies, coarser high). Output: 80 numbers per frame.
3. **dB**: convert power to decibels (`10·log10`), the usual loudness scale.

Result: **80 numbers every 20 ms (50 frames/second)**. Our cache stores 13.7 h of LibriSpeech
speech this way (3482 utterances, 38 speakers), each file `(n_frames, 80)` in float16.

Why this input: it is small (80× cheaper than raw audio), it is exactly what the frozen USM
reference model consumes (so comparisons are apples-to-apples), and 20 ms is short enough that
speech content changes only slightly frame to frame.

---

## 2. Encoder: audio → latents

`Encoder` in `model.py`. Input `(batch, T, 80)`, output `(batch, T, 256)`.

**2.1 Causal patch embedding.** `Conv1d(80 → 256, kernel 4)` over the time axis, with 3 zeros
prepended on the left. Each output frame sees the 4 mel frames ending *at* its own position.
The left-padding is the whole point: the model must never see the future, otherwise the
prediction task is unanswerable-but-trivial ("cheat by peeking"). This is the audio analogue
of a causal attention mask.

**2.2 Six delta-net blocks** (`DeltaBlock` → `DeltaCore`). This is the unusual part; the rest
is standard. Each block keeps, per attention head, a small square **memory matrix** `S`
(64×64 per head, 4 heads). At every frame `t`:

```
b_t  = sigmoid(W_b · x_t)          # write gate, a number in (0,1) per head
k_t  = normalize(W_k-conv output)  # "key": what am I writing about?
v_t  = W_v-conv output             # "value": what am I writing down?
S_t  = S_{t-1} − b_t (S_{t-1} k_t − v_t) k_tᵀ      # the delta rule
y_t  = S_t q_t                     # "query": what do I read out?
```

Read the recurrence informally: *look up what the memory currently says about this key
(`S_{t-1}k_t`), see how far that is from the value I want to store (`v_t`), and correct the
memory by exactly that error term.* "Delta" means **error-correcting write** — like editing a
notebook entry by rewriting only the part that is wrong, instead of appending a new page.
`b_t` controls how strongly to write; because it is produced from the input by a learned
linear layer plus a sigmoid, the network decides per frame and per head how much to overwrite.

Two details that are load-bearing:

* `k_t` and `q_t` are **L2-normalized**. The write magnitude is `b_t·|k_t|²`; with `|k| ≤ 1`
  the update is a contraction, which is what keeps the memory numerically stable over
  thousands of frames. (An earlier version without this produced NaN gradients.)
* There is **no explicit decay factor** `a` multiplying `S`. Forgetting is achieved by
  overwriting (`b_t` close to 1 rewrites the entry for that key). We tested adding a learned
  scalar decay; we could not find a closed-form chunk-parallel implementation for it, and the
  chunkwise algorithm we use below is exact only for the decay-free form. Beta-gate forgetting
  is the mechanism we kept, and it is what the reference implementation (Gated DeltaNet,
  arXiv:2406.06484) is built around.

**2.3 Two GQA blocks** (`GQABlock`). Ordinary causal attention with two twists: **grouped-query
attention** — 4 query heads share 2 key/value heads, saving memory bandwidth — and **RoPE**
(rotary position embeddings), which encodes *relative* position by rotating q/k vectors by an
angle proportional to the frame index. Each attention block ends with a feed-forward MLP, and
its output path is gated (`· sigmoid(W_gate x)`), a standard trick to let the network attenuate
a block's contribution.

Why this 6:2 mix: delta blocks give a cheap, constant-memory, long-range recurrence (their cost
per frame does not grow with history length), while attention can do exact content lookup
anywhere in the window — expensive, so we use only two. This ratio is the design decision from
`DESIGN_v0.md`, unchanged.

**2.4 Output norm.** A final RMSNorm (normalize each 256-vector to unit RMS, learnable scale)
gives the latent `z_t`. 12.5 M parameters total.

---

## 3. Predictor: latents → future latents

`Predictor` in `model.py`. Takes the latent stream, projects it, runs **two delta-net blocks
(no MLPs)**, and ends with **four linear heads**, each predicting the latent at a fixed offset:

| head | offset | in seconds (50 fps) |
|---|---|---|
| 1 | +1 frame | +20 ms |
| 2 | +10 frames | +200 ms |
| 3 | +50 frames | +1 s |
| 4 | +150 frames | +3 s |

Four horizons rather than one because the "correct" temporal scale for speech is unknown: a
model that can only see 20 ms ahead learns different structure than one reaching 3 s. The
multi-horizon loss forces the latent to hold information useful at several scales at once.

---

## 4. The objective

```
L = mean over horizons d of  MSE( pred_d(t),  z(t+d) )   +   α · SIGREG(z)
```

* **Targets are the encoder's own latents** — there are no human labels anywhere in this
  training. `z(t+d)` is computed by the same encoder (same weights, this same forward pass).
* We deliberately use **no EMA** (exponential moving average of the encoder, the usual JEPA
  trick to avoid the target moving too fast) and **no stop-gradient**. Those are stabilizers we
  chose not to need; SIGREG is the sole anti-collapse mechanism. This is a design decision, and
  it is the reason SIGREG's calibration matters so much below.

---

## 5. Collapse — what it is, why it happens, how we detect it

The loss above has a trivial solution. Suppose the encoder maps every input to (almost) the
same vector `c`. Then predicting the future is free: `pred = c ≈ z(t+d)` for every frame,
MSE ≈ 0, and the loss looks perfect — while the encoder carries **zero information**. This is
"representation collapse", and it is the main way self-supervised representation learning
fails silently. Our gate-4 run demonstrated it concretely: prediction loss fell to `1e-4`
across *all* horizons, and every diagnostic went dead.

Diagnostics used (all in `board.py`, all computed on a batch of latents `Z`, shape `(N, 256)`):

* **Per-dim variance** `var(z_i)` for each of the 256 coordinates, reported as
  `[min, mean, max]`. A live representation has most coordinates varying; collapse drives
  min and mean toward 0.
* **Effective rank** — participation ratio of the covariance eigenvalue spectrum,
  `(Σλ)² / Σλ²`. Intuitively, "how many independent directions actually carry variance". A
  healthy 256-d Gaussian gives ≈256; a rank-1 representation gives ≈1.
* **Speaker probe** — a linear classifier trained on frozen latents to predict the 38
  LibriSpeech speakers. Collapse makes this impossible (0.000 observed), because speaker
  identity is exactly the kind of information that gets discarded when everything maps to the
  same vector.

---

## 6. SIGREG, in plain terms

SIGREG (from UniJEPA) is a penalty that pushes the *distribution of latents over a batch*
toward a standard Gaussian — mean 0, variance 1, in every direction:

1. Sample ~512 random unit directions `u` in 256-d space.
2. For each `u`, project all latents in the batch: `p = Z·u` (one number per frame).
3. Penalize the batch mean of `p` (push toward 0) and its variance (push toward 1):
   `R = mean_u[ mean(p)² + (var(p) − 1)² ]`.

If latents cluster at a point, `var(p)` ≈ 0 for every direction and `R` is large. To make `R`
small the encoder must spread its outputs out and keep them roughly isotropic — a "filled
ball" rather than a line or a dot. **α is the weight on this penalty**: how hard we push
relative to the prediction loss.

**Why α = 0.01 was wrong and α = 1.0 is what we use.** With α = 0.01 the penalty is negligible
next to the prediction loss; on a single fixed batch (gate 1) that is fine — memorizing one
batch is the goal there — but on the real, varied dataset it collapsed (variance → 0,
eff-rank → ~3, speaker probe 0.000). With α = 1.0 the penalty actually constrains the geometry,
and the collapse stops.

---

## 7. The isotropy term — the second-order fix

α = 1.0 was necessary but not sufficient. We found a subtler failure: `R` is an **average over
directions**, so it can be small while the variance is concentrated in a handful of coordinates
and the rest of the space is dead. Analogy: a 40-piece orchestra where two violins play
everything and 38 players sit silent — total loudness (the average) looks fine, the music does
not. Observed trajectory at α = 1.0: effective rank decayed 11 → 6 → 4–5 over 300–1200 steps
while per-dim variance *mean* stayed ≈ 0.8 and `R` stayed ≈ 0.25, i.e. the regularizer reported
itself satisfied the whole time. The speaker probe read 0.000, because the information had
migrated into the dying coordinates.

Fix (implemented, switchable): add a term that penalizes **the spread of projection variance
across directions**,

```
R_iso = R  +  λ · Var_u[ var_z(u·z) ]
```

which is zero only if all directions share the same variance (isotropic) and grows when a few
directions hog the variance. Control run at λ = 1.0, α = 1.0: effective rank *floored* at ≈ 7.5
instead of decaying, minimum per-dim variance held at 0.12–0.18 instead of 0, and a window-level
probe read 0.987. Pre-registered bar of ≥ 15 was not met — the term works in the right
direction but is under-scaled at λ = 1; λ = 4 and a kurtosis term are the next dials. Defaults
in the code are `isotropy = 0.0` so existing gate behaviour is unchanged.

---

## 8. Training recipe as run

| item | value |
|---|---|
| data | 13.7 h LibriSpeech train-clean-100, 80-mel @ 50 fps cache (3482 utts, 38 speakers) |
| sampling | random utterance, random offset, window T frames (250 = 5 s for gates; 500 = 10 s for the long run) |
| batch | 8 windows |
| optimizer | AdamW, lr 3e-4, weight decay 0, grad-norm clip 1.0 |
| objective | multi-horizon MSE + α·SIGREG; α = 1.0, isotropy λ = 1.0 |
| precision | fp32 on CPU (bf16 autocast measured **19× slower** on this CPU); bf16 on MPS |
| hardware | M1, 8 GB unified; measured peak ≈ 1.0 GiB at bs 8 / T 250 |

**The gate ladder** (each gate must pass before the next; all in `train.py`):

1. **Overfit one batch** → loss must fall to ≈ 0. Catches wiring/optimizer errors.
2. **Seeded determinism** → two fresh runs must give bit-identical step-0 losses. Catches
   hidden state and data-order bugs.
3. **Collapse dial** → α = 0 must collapse (variance → 0); α > 0 must not. Proves the
   regularizer is actually connected to the geometry.
4. **Real run** → full cache, board rows every 100 steps, checkpoint. This is where collapse
   was discovered at α = 0.01, and where the isotropy fix was validated.

**Six-check smoke test** (`python model.py`): determinism, shape contract over B/T, gradient
flow + gate spread, BIBO stability (state norm must shrink when input goes to zeros), a
hand-computed known-answer test for the delta rule (T = 2 gives 0.5 then 0.75 — the geometric
convergence the delta rule predicts), and causality (perturbing frames ≥ t must not change
outputs < t).

---

## 9. The scan: chunkwise, and now log-depth parallel

The recurrence for `S` is sequential in time, which is slow if implemented naively (one Python
iteration per frame). Two accelerations are in use:

**Chunkwise (UT transform, Yang et al. 2406.06484 Eqs. 5–11).** Split the window into chunks of
64 frames. Inside a chunk, the delta rule can be written as one triangular linear solve
(`(I + A)⁻¹`, where `A` is the strictly-lower-triangular matrix of `b_i·(k_i·k_j)`) instead of
64 sequential updates; across chunks, one matrix carry is propagated. Python-level iterations
drop from `T` to `T/64`. Verified against the per-step reference: outputs and final states
match to < 1e-5.

**True parallel scan (replacing the remaining chunk loop).** The chunk carry is *affine* in the
state: `S_out = S_in·M_c + b_c` with `M_c = I − W_cᵀK_c`, `b_c = U_cᵀK_c`, all computable for
every chunk independently. Affine maps compose associatively —
`(S·M₁ + b₁)·M₂ + b₂ = S·(M₁M₂) + (b₁M₂ + b₂)` — so the prefix composition needed to know each
chunk's incoming state can be computed with a **Hillis–Steele doubling scan**: log₂(N_chunks)
parallel rounds instead of N_chunks sequential steps (this is the same idea as Mamba's SSD
parallel scan). Sequential depth goes from `T/64` to `log₂(T/64)`; for a 5 s window that is
4 sequential steps → 2. This matters most on MPS, where each sequential launch costs fixed
overhead.

**Measured verdict (15 Oct, 16:15 clean re-run — this is the part that matters, and it is weaker than the argument
above suggests).** Verified against the per-step reference three independent ways (≤ 7.2e-07) and
the depth claim holds (8 → 3 at T = 500). But wall-clock: **≤ 7% on MPS, ≈ 0% on CPU, and −10% at
T = 500**, because per-frame cost is *flat in T in both* variants — the step is dominated by
memory-bound elementwise kernels and the SIGREG projections, not by chunk-loop launches. Op count
actually **rises** (78 830 → 88 328 at T = 100, +12%), since the scan trades few big steps for many
small views (`as_strided`/`select`/`resolve_conj`). So the honest label is **"more parallel, not
yet faster"**: depth is real and pays where serial steps are the binding cost (a bare cell at large
T, under `torch.compile`/graph capture) — which is untested here. Keep it; do not claim a speedup.
Full numbers: `POST_MORTEM.md` §1.1, `references/perf-and-measurement-discipline.md`.

---

## 10. Evidence that it works (what "actually works" means here)

Because the model's native output is a vector, "works" is demonstrated with three artifacts:

1. **Linear probes on frozen latents** (no fine-tuning of the encoder — a linear classifier on
   top): speaker identification (38 classes; blind floor = majority class) and a coarse
   acoustic-semantics task, speech vs music, where the music side is *synthetic* (generated
   chords/sweeps through the identical mel pipeline) and labelled as such.
2. **Lightweight decoder**: latents → mel → Griffin–Lim → audible waveform. Griffin–Lim recovers
   a signal from a magnitude spectrogram by iteratively estimating phase; it is rough but
   intelligible. This is the piece that turns latents into something you can listen to.
3. **Visualization of model state**: PCA of pooled latents coloured by speaker, plus delta-net
   state norms over time for speech vs music input — a direct look at what the recurrence
   does.

Plus an inference CLI: give it a `.wav` (any sample rate) **or** a text string (synthesized
locally with macOS `say`, since the design contains no text tokenizer — audio is the only
input modality) and it returns: probe tags, nearest-neighbour training utterance, speaker
guess with confidence, and a playable reconstruction.

### 10.1 The numbers, as measured (v0, `runs/v0`, 26 MB total)

| probe | result | floor / ceiling |
|---|---|---|
| Held-out reconstruction | 764 → **17.1 dB²** (mel) | — |
| Effective rank of latents | 8.6 → **20.8** (no collapse) | 256 max |
| Speaker ID, linear probe on FROZEN latents (38 classes, n = 599) | **0.991 ± 0.003** | blind 0.040; USM reference 0.9996 |
| Speech vs music (music = synthetic, labelled) | **1.000** | 0.500 |
| Decoder autoencode (mel corr) | **0.964** | Griffin–Lim pipeline ceiling 0.997 |
| Decoder on frozen latents (beats joint training) | **0.971** | — |
| 1 s-ahead prediction | **FAILS** — corr 0.44, 268 dB², flat | > 0.9 |

### 10.2 "Not gibberish", tested directly (`_verify_infer_spec.py`, `runs/verify_infer_spec.log`)

Round-tripping three *different* speakers through the full user path (clip → latents → decoded
wav → re-encoded → probe):

* **Identity vs mismatch.** Decoding clip A's latent fits A **16.4–28.0×** better (mel-MSE) than
  decoding clip B's does. A decoder that emitted generic "speech" would score ≈ 1×.
* **Speaker survives the loudspeaker.** 2/3 self-decodes still classify as the same speaker
  (the third was already an unsure reading, p = 0.38); cross-decodes change the speaker in 3/3.
* **It is sound, not noise.** Spectral flatness 0.0015–0.079 (white noise ≈ 1.0), RMS within 20%
  of the input.
* **Budget.** Peak RSS for the whole path, 3 clips + 6 Griffin–Lim decodes: **0.33 GB** (limit 4 GB),
  0.8 s.

Text path (`infer.py --text "…"`, `runs/verify_text_out.log`): decoded mel tracks the true TTS
audio at **corr 0.964** against a 0.997 pipeline ceiling, SNR 8.44 vs 9.73 dB — i.e. **most of the
remaining error is the Griffin–Lim vocoder, not the latents**. Tag came back `speech-like
(p_music=0.00)`, nearest training utterance cosine 0.676.

bf16 (`_verify_bf16_mps.py`, `runs/verify_bf16_mps.log`): runs on MPS, **1.0015× the fp32
reconstruction MSE, max deviation 0.38 dB, no NaN**; MPS fp32 is bit-identical to CPU fp32. bf16 is
therefore supported and numerically safe — it just buys no speed on this machine, so fp32 on CPU
stays the training default.

---

## 11. Known limits (stated plainly)

* 13.7 h of speech, one corpus, 38 speakers — small. Probes are evidence of *learned structure*,
  not benchmark-grade accuracy.
* The decoder is a demonstration path, not a vocoder; expect robotic Griffin–Lim audio. This, not
  the latents, is the current quality ceiling: the pipeline's own best is corr 0.997 against 0.964
  for the model's decode.
* **The predictor does not work yet.** 1 s-ahead is corr 0.44 — the model reconstructs and
  characterises audio well but does not yet predict its future. It emits *structured* audio that is
  the wrong content, so beware of judging it by ear or by flatness alone; the honest measures are
  mel-MSE/corr against the true future.
* Isotropy λ = 1 now holds eff-rank **20.8** at full scale (it was ≈ 7.5 in the small runs where
  λ = 4 looked like the next step) — the earlier note that the constant was under-tuned is obsolete
  for v0; λ remains worth sweeping, not fixing.
* MPS is *not* the fast path on this machine: at bs = 8 CPU fp32 is 2.8× MPS on throughput, and bf16
  (though verified numerically equivalent) buys no speed. GPU/MPS is a portability option, not a win.
* No text modality: adding one is a separate contrastive rung, not a switch to flip.
