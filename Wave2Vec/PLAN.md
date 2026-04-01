You are building a research-grade Bengali Audio Deepfake Detection system.

Implement the following full pipeline in a clean, modular Python notebook using PyTorch.

==============================
PROJECT OVERVIEW
==============================

Goal:
Detect Bengali deepfake audio (13k real + 12k VITS fake)
Architecture:
WavLM-Large + Modified AASIST + OC-Softmax + RawBoost

Dataset:
- BanglaFake dataset
- Speaker-independent split
- 70% train
- 15% validation
- 15% test

==============================
STEP 1 — ENVIRONMENT SETUP
==============================

Use:
- PyTorch
- torchaudio
- transformers
- numpy
- sklearn
- librosa
- matplotlib

Set random seeds for reproducibility.

==============================
STEP 2 — DATA PIPELINE
==============================

1. Load audio at 16kHz.
2. Apply speaker-independent stratified split.
3. Implement RawBoost augmentation including:
   - Random gain scaling
   - Additive Gaussian noise (SNR 5–30 dB)
   - Random clipping
   - Reverberation using convolution
   - MP3 compression simulation
   - Pitch shift (±1 semitone)
   - Time stretch (0.9–1.1)

Augmentation must only apply to training data.

==============================
STEP 3 — WAVLM FRONTEND
==============================

Load pretrained model:
microsoft/wavlm-large

Freeze:
- Feature extractor
- First 12 transformer layers

Fine-tune:
- Remaining transformer layers

Extract contextual embeddings for each audio segment.

==============================
STEP 4 — MODIFIED AASIST
==============================

Implement AASIST backend WITHOUT SincNet.

Input:
WavLM embeddings

Include:
- Graph attention layers
- Spectro-temporal modeling
- Residual connections
- Dropout (0.3)

Output:
128-dimensional embedding

==============================
STEP 5 — OC-SOFTMAX CLASSIFIER
==============================

Implement One-Class Softmax loss:

- Compact real samples into cluster
- Push fake samples outside margin

Use margin = 0.5
Scale factor = 10

==============================
STEP 6 — TRAINING LOOP
==============================

Optimizer:
AdamW

Learning rates:
1e-5 for WavLM
1e-4 for AASIST

Train 50 epochs with:
- Early stopping based on validation EER
- Gradient clipping (1.0)

Track:
- Loss
- Accuracy
- EER
- ROC-AUC

==============================
STEP 7 — EVALUATION
==============================

On test set compute:
- Equal Error Rate (EER)
- ROC curve
- DET curve
- Accuracy
- Precision
- Recall
- F1 score

Plot:
- Confusion matrix
- ROC curve
- DET curve

==============================
STEP 8 — ROBUSTNESS TESTING
==============================

Evaluate under:
- 10 dB noise
- MP3 compression
- Reverberation

Report performance drop.

==============================
STEP 9 — EXPLAINABILITY
==============================

Implement:
- Attention visualization from AASIST
- Frequency band importance using SHAP (optional)

==============================
OUTPUT
==============================

Notebook must:
- Be clean and modular
- Include training and inference functions
- Save best model
- Save evaluation metrics
- Be reproducible

Return complete working code.
