# Audio Semantic Predictor: Project Plan

## 1. Core Idea

Build a modality-agnostic semantic prediction system from audio.
The model learns to predict future semantic embeddings in a shared representation space, rather than reconstructing specific tokens, waveforms, or any single modality.
These embeddings can later be decoded into arbitrary downstream modalities: audio waveforms, text transcriptions, visual features, or other representations.
Goal :
Learn stable, temporally coherent semantic states that capture "what is happening" in the audio stream—speech content, speaker characteristics, prosody, musical structure, environmental sounds—purely through self-supervised prediction of future states.
Key principle :
The semantic space is learned by forecasting, not by reconstruction.
This forces the model to extract high-level meaning rather than memorize surface patterns.

## 2. Architecture Overview

### 2.1 Input Processing

- Input modality :
Raw audio converted to log-Mel spectrograms

- Preprocessing : 24 kHz sampling rate, mono channel, fixed hop size

- Training clips :
Long clips (e.g., 10 seconds) split into time-stepped frames

### 2.2 Dual Encoder System (JEPA-style with State) Teacher Encoder

- Observes audio from time to (full context plus future chunk) 𝑡𝑡+𝑤+𝑘

- Maintains continuous hidden state across time steps

- Parameters updated via Exponential Moving Average (EMA) of student weights

- Outputs target semantic embeddings 𝑧 𝑡 + 1 : 𝑡 + 𝑘𝑡𝑒𝑎𝑐ℎ𝑒𝑟

- Provides stable, history-aware targets for training Student Encoder

- Observes audio only from time to (no future access) 𝑡𝑡+𝑤

- Maintains its own continuous hidden state ℎ 𝑡

- Architecture: Mamba (SSM) or similar stateful architecture

- Outputs current semantic embedding 𝑧 𝑡𝑠𝑡𝑢𝑑𝑒𝑛𝑡 = 𝑓 ( ℎ 𝑡 )

- Trained via gradient descent to match teacher's future predictions

### 2.3 Predictor Module The predictor is a stateless forecasting module that maps current representations to future semantic states:

- Input: Current student embedding and time offset 𝑧 𝑡𝑠𝑡𝑢𝑑𝑒𝑛𝑡 ∆

- Output: Predicted future embedding 𝑧ˆ 𝑡 + ∆

- Architecture: MLP or small residual network

- Loss: Compares with teacher's 𝑧ˆ 𝑡 + ∆ 𝑧 𝑡 + ∆𝑡𝑒𝑎𝑐ℎ𝑒𝑟 o Options: cosine similarity loss, InfoNCE contrastive loss, or L2 regression

### 2.4 Decoder Heads (Modality-Specific, Trained Later) The semantic backbone is modality-agnostic.
Decoders are thin task-specific heads trained after the predictor backbone is established:
Audio Decoder

- Input: Sequence of semantic embeddings over time

- Output: Waveform or mel-spectrogram

- Architecture: HiFi-GAN, Vocos, or convolutional decoder

- Loss: Reconstruction loss + perceptual/adversarial losses Text Decoder (Optional, Downstream)

- Input: Compressed sequence of semantic embeddings

- Output: Text tokens (ASR or captioning)

- Architecture: Small transformer decoder

- Loss: Cross-entropy on text labels Visual Decoder (Optional, Downstream)

- Input: Audio semantic embeddings aligned with video timestamps

- Output: Frame-level visual features or captions

- Loss: Alignment loss (e.g., contrastive video-audio matching) All decoders share the same underlying semantic embedding space, enabling true multimodal capability from a unimodal pretraining foundation.

## 3. Training Procedure

Phase 1:
Semantic Predictor Pretraining Step 1:
Data Preparation 1.
Sample long audio clips from diverse datasets (speech, music, environmental noise) 2.
Convert to log-Mel spectrograms at 24 kHz, mono channel 3.
Split into training clips with metadata for downstream tasks Step 2:
Forward Pass (Student) 1.
Input frames from time to into student encoder 0𝑇 2.
Recurrently update student hidden state (Mamba SSM architecture) ℎ 𝑡 3.
At each time step, produce semantic embedding 𝑧 𝑡𝑠𝑡𝑢𝑑𝑒𝑛𝑡 Step 3:
Forward Pass (Teacher) 1.
Input same audio, but teacher sees extended window including future frames 2.
Update teacher state and produce embeddings for all time steps 𝑧 𝑡𝑡𝑒𝑎𝑐ℎ𝑒𝑟 3.
Teacher weights remain frozen during forward pass (EMA-updated only) Step 4:
Prediction and Loss Computation For each anchor time and future offsets : 𝑡∆∈{1,2,...,𝑘} 1.
Predictor takes and offset 𝑧 𝑡𝑠𝑡𝑢𝑑𝑒𝑛𝑡 ∆ 2.
Outputs predicted future embedding 𝑧ˆ 𝑡 + ∆ 3.
Compute loss between and 𝑧ˆ 𝑡 + ∆ 𝑧 𝑡 + ∆𝑡𝑒𝑎𝑐ℎ𝑒𝑟 𝐿 𝑝𝑟𝑒𝑑 = ∆ = 1𝑘 ∑ 𝑙𝑜𝑠𝑠 ( 𝑧 ˆ 𝑡 + ∆ , 𝑧 𝑡 + ∆ 𝑡𝑒𝑎𝑐ℎ𝑒𝑟 ) # ( 1 ) Common loss choices: ● Cosine similarity: 1 − 𝑐𝑜𝑠 ( 𝑧 ˆ , 𝑧𝑡𝑒𝑎𝑐ℎ𝑒𝑟 ) ● InfoNCE contrastive loss ● Mean squared error (L2 regression) Step 5:
Backpropagation and Updates 1.
Backpropagate through student encoder and predictor only 2.
Update student encoder weights via gradient descent 3.
Update teacher encoder weights via EMA: θ 𝑡𝑒𝑎𝑐ℎ𝑒𝑟←τθ 𝑡𝑒𝑎𝑐ℎ𝑒𝑟+(1−τ)θ 𝑠𝑡𝑢𝑑𝑒𝑛𝑡 4.
Typical τ∈[0.99,0.999] Masking Strategy (optional, inspired by JEPA): ● Mask certain time regions in student input ● Force predictor to infer masked future regions from unmasked context ● Prevents trivial solutions and encourages semantic compression Result :
A stateful encoder whose latent space is trained to forecast future semantic content, not reconstruct input patterns.
Phase 2:
Decoder Training (Modality-Specific Heads) Once the semantic predictor backbone achieves stable training (convergence on prediction loss), attach and train decoders:
Audio Reconstruction Head 1.
Freeze or lightly fine-tune the semantic encoder backbone 2.
Input:
Full sequence of semantic embeddings 𝑧 1 : 𝑇 3.
Train decoder to reconstruct original audio (waveform or spectrogram) 4.
Objective:
Reconstruction loss + perceptual loss (e.g., mel-spectrogram distance) + adversarial loss (optional) Text Head (ASR / Captioning) 1.
Input:
Downsampled or compressed semantic embedding sequence 2.
Train small transformer decoder to generate text tokens 3.
Objective:
Cross-entropy loss on ground-truth transcriptions or captions

## 4. Data and Preprocessing

4.1 Datasets Category Dataset Target Size Purpose Speech LibriSpeech (train-clean-100) $\sim$30 GB Clean, continuous sentences; structural backbone Music FMA Small (Free Music Archive) $\sim$8 GB 8 genres; frequency richness and musical structure Noise ESC-50 or AudioSet (balanced) $\sim$600 MB 50 classes of natural/urban sounds; prevents hallucination Table 1:
Training datasets covering speech, music, and environmental audio Why these datasets: ● Speech (LibriSpeech) :
Provides clean, linguistically structured audio for learning temporal coherence and phonetic patterns ● Music (FMA Small) :
Teaches frequency richness, harmonic structure, and long-range temporal dependencies ● Noise (ESC-50/AudioSet) :
Adds environmental diversity, prevents overfitting to speech/music, enforces robust semantic representations 4.2 Preprocessing Pipeline 1.
Resampling :
Convert all audio to 24 kHz sampling rate 2.
Channel normalization :
Force mono channels (average stereo if needed) 3.
Segmentation :
Split long recordings into fixed-length clips (e.g., 10-30 seconds) 4.
Spectrogram computation :
Compute log-Mel spectrograms with fixed parameters o Window size: 1024 samples (42.7 ms at 24 kHz) o Hop length: 256 samples (10.7 ms) o Mel bins: 80-128 5.
Normalization :
Per-clip or global mean-variance normalization 6.
Metadata preservation :
Track clip ID, timestamps, and labels for downstream decoder training 4.3 Data Augmentation (Optional) ● Time stretching (0.9x - 1.1x) ● Pitch shifting (±2 semitones) ● Background noise mixing (SNR 10-30 dB) ● SpecAugment (frequency/time masking)

## 5. Implementation Details

5.1 Model Hyperparameters Student Encoder (Mamba SSM) ● Hidden dimension: 512-1024 ● Number of layers: 8-12 ● State dimension: 64-128 ● Output embedding dimension: 256-512 Teacher Encoder ● Same architecture as student ● EMA coefficient : 0.996-0.999 τ ● Updated every training step Predictor ● Architecture: 2-3 layer MLP or residual network ● Hidden dimension: 512 ● Output dimension: matches embedding dimension ● Activation:
GELU or SiLU Decoders (Phase 2) ● Audio decoder:
HiFi-GAN or similar vocoder architecture ● Text decoder: 6-layer transformer decoder, vocab size 50k-100k ● Visual decoder:
Projection MLP + contrastive head 5.2 Training Configuration Phase 1:
Semantic Predictor ● Batch size: 32-64 clips ● Sequence length: 1000-3000 frames (depends on clip duration) ● Optimizer:
AdamW, learning rate with cosine decay 3 × 10−4 ● Warmup: 5000 steps ● Gradient clipping: max norm 1.0 ● Training steps: 500k-1M ● Prediction horizon : 8-16 future steps 𝑘 ● Hardware: 4-8 GPUs (A100 or H100 recommended) Phase 2:
Decoder Training ● Freeze or lightly fine-tune backbone (learning rate for backbone, for 10−5 10−4 decoder) ● Batch size: 16-32 ● Optimizer:
AdamW ● Training steps: 100k-200k per decoder ● Mixed precision training (FP16 or BF16) 5.3 Monitoring and Evaluation Semantic predictor metrics (Phase 1) ● Prediction loss over time (should decrease and stabilize) ● Cosine similarity between predicted and target embeddings ● Embedding space visualization (t-SNE or UMAP) ● Temporal coherence metric (autocorrelation of embeddings) Decoder metrics (Phase 2) ● Audio reconstruction:
PESQ, STOI, mel-spectrogram distance ● Text decoder:
WER (Word Error Rate), CER (Character Error Rate) ● Visual decoder:
Retrieval accuracy (audio-to-video and video-to-audio) Downstream evaluation ● Speech recognition benchmarks (LibriSpeech test-clean, test-other) ● Music genre classification (GTZAN, FMA validation set) ● Audio event detection (ESC-50, AudioSet evaluation) ● Zero-shot transfer to new audio domains

## 6. Expected Outcomes

6.1 Semantic Embedding Properties ● Temporal coherence :
Smooth transitions between consecutive embeddings ● Semantic richness :
Embeddings capture phonetic, prosodic, musical, and environmental cues ● Modality-agnostic :
Same embedding space works for audio, text, and visual tasks ● Predictive power :
Embeddings enable accurate forecasting of future audio content 6.2 Performance Targets ● Audio reconstruction :
Near-lossless quality (PESQ > 4.0) ● Speech recognition :
Competitive with supervised baselines (WER < 5% on LibriSpeech test-clean) ● Music understanding :
Genre classification accuracy > 85% ● Robustness :
Stable performance on noisy, out-of-domain audio 6.3 Scalability and Transfer ● Pretrained backbone enables quick fine-tuning for new tasks (< 10k labeled examples) ● Embeddings generalize to unseen audio domains without retraining ● Modular decoder design allows adding new modalities without retraining backbone

## 7. Practical Next Steps

7.1 Prototyping Phase (Weeks 1-2) 1.
Implement minimal student-teacher encoder with 2-3 time step prediction 2.
Test on small subset of LibriSpeech (1 hour of audio) 3.
Verify EMA update mechanism and gradient flow 4.
Validate that prediction loss decreases 7.2 Scaling Phase (Weeks 3-6) 1.
Scale to full LibriSpeech train-clean-100 dataset 2.
Add FMA Small and ESC-50 for diversity 3.
Implement efficient data loading pipeline (streaming, prefetching) 4.
Monitor training stability (gradient norms, loss curves) 5.
Use learning rate warmup and gradient clipping 7.3 Decoder Training Phase (Weeks 7-10) 1.
Freeze semantic predictor backbone 2.
Train audio reconstruction decoder first (easiest to debug) 3.
Train text decoder on LibriSpeech transcriptions 4.
Optionally train visual decoder if audio-video data is available 5.
Evaluate each decoder independently 7.4 Evaluation and Iteration (Weeks 11-12) 1.
Compare with baselines (supervised ASR, music classification models) 2.
Perform ablation studies (number of prediction steps, EMA coefficient, loss type) 3.
Analyze embedding space (clustering, similarity metrics) 4.
Test zero-shot transfer to new audio domains 5.
Document findings and prepare for scaling or deployment

## 8. Key Design Choices and Rationale

8.1 Why Dual Encoders (Student-Teacher)? ● Stability :
Teacher provides stable targets via EMA, preventing training collapse ● Self-supervised :
No need for labels; model learns from temporal structure alone ● Representation quality :
Forces student to compress information efficiently 8.2 Why Predict Embeddings, Not Audio? ● Semantic focus :
Predicting waveforms encourages memorization of low-level patterns (phase, noise) ● Modality flexibility :
Semantic embeddings can map to any output modality ● Compression :
Embeddings are compact, high-level representations 8.3 Why State-Based Encoders (Mamba/SSM)? ● Long-range dependencies :
Audio requires modeling dependencies over seconds or minutes ● Efficiency :
Linear-time complexity in sequence length (unlike quadratic attention) ● Recurrent structure :
Natural fit for streaming audio and temporal coherence 8.4 Why Multiple Data Sources (Speech, Music, Noise)? ● Generalization :
Single-domain training leads to overfitting and domain-specific artifacts ● Robustness :
Diverse audio teaches model to ignore irrelevant variations ● Completeness :
Real-world audio contains mixtures of speech, music, and environmental sounds

## 9. Risks and Mitigation Strategies

Risk Mitigation Strategy Training instability (collapse) Use EMA for teacher, gradient clipping, warmup Overfitting to one dataset Mix speech, music, noise; use data augmentation Computational cost Mixed precision (FP16), efficient data loading, gradient checkpointing Decoder quality poor Pretrain backbone thoroughly; use perceptual losses Embeddings not semantic Increase prediction horizon ; use 𝑘 contrastive loss Table 2:
Risk mitigation strategies for training

## 10. Conclusion

This project proposes a novel approach to learning audio representations through semantic prediction rather than reconstruction.
By training a dual-encoder system to forecast future embeddings in a shared semantic space, we obtain modality-agnostic representations that can be decoded into audio, text, or visual outputs.
The phased training approach—first establishing a strong semantic predictor backbone, then training lightweight modality-specific decoders—ensures modularity, scalability, and transfer learning capability.
The use of diverse audio data (speech, music, environmental sounds) guarantees robustness and generalization.
Core innovation :
Predictive self-supervision on semantic embeddings, not on raw signals, forces the model to learn high-level meaning rather than surface patterns.
This enables true multimodal understanding from a unimodal (audio) pretraining foundation.
Next immediate action :
Implement minimal prototype with 2-3 time step prediction on LibriSpeech subset to validate architecture and training dynamics.
