# Final Presentation Plan

We will document our final presentation in this file. We will include the following sections:

- Problem definition
- Dataset and its analysis (Statistics)
- Proposed solution (architecture)
- Loss function and its intuition.
- Performance report
- Comparison with state-of-the-art methods
- Challenges/Discussions

The section below provides a summary of what has been done in the project, including the datasets used, the models implemented, and the results obtained.

# Summary of the Project

**We dealt with the following datasets in our project** :

# 1. LSTM Bangla Audio Dataset:

This dataset was used for the replication study of the original paper `Bangla Deepfake Detection Using LSTM and Wavenet by Ayan et al. (2026)`. It contains both real and deepfake audio samples in Bangla.

- Statistics:
  - Total samples: 4500
  - Real audio samples: 2,250
  - Deepfake audio samples: 2,250

**Notebook 1** : [original_bangla_deepfake_detection.ipynb](../Inference/Checkpoint%203/original_bangla_deepfake_detection.ipynb) notebook was ran on this dataset.

### Result

| Metric    | Real   | Fake   | Weighted Avg |
| --------- | ------ | ------ | ------------ |
| Precision | 0.9956 | 0.9978 | 0.9967       |
| Recall    | 0.9978 | 0.9956 | 0.9967       |
| F1-Score  | 0.9967 | 0.9967 | 0.9967       |
| Support   | 450    | 450    | 900          |

**Overall Accuracy: 0.9967**

| Model        | Accuracy | Precision | Recall | F1 Score |
| ------------ | -------- | --------- | ------ | -------- |
| LSTM (Paper) | 0.8900   | 0.8906    | 0.8898 | 0.8967   |
| LSTM (Ours)  | 0.9967   | 0.9967    | 0.9967 | 0.9967   |

![Paper lstm results vs our lstm results](./images/paper-lstm-vs-our-lstm.png)

## Findings:

- key weaknesses:

1. MFCC loses high-frequency artifacts
2. LSTM struggles with:
   - long sequences
   - subtle artifacts
   - Poor generalization to unseen TTS (Gemini, Crikk)

# 2. BanglaFake Dataset :

- Statistics:
  - Total samples: 26592
  - Real audio samples: 13796
  - Deepfake audio samples: 12796

**Notebook 2**: [lstm-pipeline.ipynb](../Inference/Checkpoint%201/lstm-pipeline.ipynb) notebook was ran on this dataset.

## What this notebook does:

**Replication Study** — Based on the paper by Ayan et al. (2026)

This notebook replicates the methodology from the paper using Bangla Fake dataset.
We implement two models:

1. **RNN-based LSTM** using MFCC features
2. **Custom WaveNet** using normalized raw audio waveforms

### Idea behind the notebook

This notebook is a more advanced replication + extension:

It implements:

- LSTM model (MFCC-based)
- WaveNet-style model (raw audio-based)

👉 In short:
**Comparison of feature-based vs raw-audio deep learning**

**Why add WaveNet (or CNN on raw audio)?**

MFCC problem:

It removes:

- phase information
- high-frequency artifacts

WaveNet idea:

Learn directly from raw waveform → no information loss

### Conceptual comparison

| Approach                  | Idea                         | Limitation      |
| ------------------------- | ---------------------------- | --------------- |
| MFCC + LSTM               | Human-designed features      | Loses artifacts |
| Raw Audio + CNN (WaveNet) | Learn features automatically | Needs more data |

**Hypothesis of this notebook**

- If deepfake artifacts exist in raw signal,
  then learned filters (CNN/WaveNet) will detect them better than MFCC.

---

## Results:

| Model   | Precision (Real) | Recall (Real) | F1-Score (Real) | Precision (Fake) | Recall (Fake) | F1-Score (Fake) | Overall Accuracy |
| ------- | ---------------- | ------------- | --------------- | ---------------- | ------------- | --------------- | ---------------- |
| LSTM    | 0.9929           | 0.9809        | 0.9868          | 0.9811           | 0.9930        | 0.9870          | 0.9869           |
| WaveNet | 0.5000           | 1.0000        | 0.6667          | 0.0000           | 0.0000        | 0.0000          | 0.5000           |

**Comparison with Paper:**

| Model           | Accuracy | Precision | Recall | F1 Score |
| --------------- | -------- | --------- | ------ | -------- |
| LSTM (Paper)    | 0.8900   | 0.8906    | 0.8898 | 0.8967   |
| LSTM (Ours)     | 0.9869   | 0.9870    | 0.9869 | 0.9869   |
| WaveNet (Paper) | 0.9100   | 0.9100    | 0.9100 | 0.9100   |
| WaveNet (Ours)  | 0.5000   | 0.2500    | 0.5000 | 0.3333   |

But unfortunately, our WaveNet implementation performed very poorly on the BanglaFake dataset, achieving only 50% accuracy. This could be due to several reasons such as insufficient training data for the WaveNet model, suboptimal hyperparameters, or the complexity of the model architecture.

Since wavenet performs poorly, we decided to explore other raw audio models like `RawNet2`, `AASIST`, and `XLSR` in the next notebook.

At this point, we discovered another limitation. When we generated some fake samples using advance TTS models (Gemini and Crikk), our LSTM model trained on the BanglaFake dataset failed to generalize to these new types of deepfake audio, achieving very low accuracy. This highlighted a key weakness of our approach: The BanglaFake dataset was generated using a specific VITS algorithm and so our LSTM model was learning to detect artifacts specific to that algorithm, rather than learning more generalizable features of deepfake audio. This motivated us to create a new dataset of TTS-generated fake audio samples to further test the generalization capabilities of our models.

# 2.1 Advanced Raw Audio Model on BanglaFake Dataset

This section describes the implementation of a state-of-the-art deepfake detection pipeline using WavLM-Large as the frontend and a modified AASIST backend, trained on the BanglaFake dataset.

- Statistics:
  - Total samples: 26592
  - Real audio samples: 13796
  - Deepfake audio samples: 12796

**Notebook**: [wavlm_aasist_pipeline.ipynb](../Inference/Checkpoint%203/wavlm_aasist_pipeline.ipynb) notebook was run on this dataset.

## What this notebook does:

This notebook implements a complete pipeline for detecting deepfake Bangla audio using advanced raw audio processing techniques.

### Components:

| Component        | Detail                                                |
| ---------------- | ----------------------------------------------------- |
| **Frontend**     | WavLM-Large (fine-tuned top 12 layers)                |
| **Backend**      | Modified AASIST with graph attention                  |
| **Loss**         | One-Class Softmax (OC-Softmax)                        |
| **Augmentation** | RawBoost (noise, reverb, compression, pitch, stretch) |

### Key Features:

- **Data Pipeline**: Speaker-independent train/val/test splits to ensure generalization.
- **RawBoost Augmentation**: Seven types of augmentation applied randomly during training for robustness.
- **Model Architecture**:
  - WavLM Frontend: Pretrained WavLM-Large with fine-tuning of top layers.
  - Modified AASIST Backend: Temporal convolutions, spectral nodes via soft attention, heterogeneous graph attention.
  - OC-Softmax Loss: Angular margin loss for better separation.
- **Training**: AdamW optimizer with differential learning rates, cosine annealing, mixed precision, early stopping.
- **Evaluation**: Comprehensive metrics including EER, ROC-AUC, accuracy, precision, recall, F1. Robustness testing under noise, compression, and reverberation.
- **Explainability**: Attention visualization to understand model focus.

## Results:

| Metric                       | Value                                                    |
| ---------------------------- | -------------------------------------------------------- |
| Architecture                 | WavLM-Large + Modified AASIST + OC-Softmax               |
| Dataset                      | 26592 samples (13796 real, 12796 fake)                   |
| Split                        | Train 19237 / Val 5461 / Test 1894 (speaker-independent) |
| Best Epoch                   | 32                                                       |
| Test EER                     | 0.1594                                                   |
| Test ROC-AUC                 | 0.9364                                                   |
| Test Accuracy                | 0.8062                                                   |
| Test Precision (Real)        | 0.9632                                                   |
| Test Recall (Real)           | 0.7210                                                   |
| Test F1 (Real)               | 0.8247                                                   |
| Test Precision (Fake)        | 0.6653                                                   |
| Test Recall (Fake)           | 0.9527                                                   |
| Test F1 (Fake)               | 0.7835                                                   |
| Robustness 10dB Noise EER    | 0.1837                                                   |
| Robustness MP3 Compress. EER | 0.2223                                                   |
| Robustness Reverb (0.5s) EER | 0.1806                                                   |

## Result Plots:

![Training Progress](./images/bangla-fake/wavlm-assist.png)

![EER , AUC and Acc Under robust conditions](./images/bangla-fake/acc-auc-eer-under-different-condition.png)

We visualized the attention weights from the modified AASIST backend to understand which parts of the audio the model focuses on when making its predictions. The attention visualization shows that the model attends to specific temporal and spectral regions of the audio signal, which may correspond to artifacts introduced by deepfake generation.

![Attention Visualization](./images/bangla-fake/attention-heads.png)

## Findings:

- The model achieves an EER of 15.94% and ROC-AUC of 93.64%, showing good performance on the BanglaFake dataset.
- Robustness testing shows performance degradation under noise (EER 18.37%), MP3 compression (22.23%), and reverberation (18.06%), indicating areas for improvement.
- The use of WavLM frontend and AASIST backend provides a strong foundation for raw audio deepfake detection.
- Attention visualization helps in understanding which audio regions are important for classification.

## Summary of Models that are trained and evaluated on BanglaFake Dataset

| Model               | Accuracy | EER    | ROC-AUC | Precision (Real/Fake) | Recall (Real/Fake) | F1 (Real/Fake)  | Robustness EER (Noise/MP3/Reverb) |
| ------------------- | -------- | ------ | ------- | --------------------- | ------------------ | --------------- | --------------------------------- |
| LSTM (MFCC-based)   | 0.9869   | N/A    | N/A     | 0.9929 / 0.9811       | 0.9809 / 0.9930    | 0.9868 / 0.9870 | N/A                               |
| WaveNet (Raw Audio) | 0.5000   | N/A    | N/A     | 0.5000 / 0.0000       | 1.0000 / 0.0000    | 0.6667 / 0.0000 | N/A                               |
| WavLM + AASIST      | 0.8062   | 0.1594 | 0.9364  | 0.9632 / 0.6653       | 0.7210 / 0.9527    | 0.8247 / 0.7835 | 0.1837 / 0.2223 / 0.1806          |

We wanated to prove that the LSTM model trained on the BanglaFake dataset is not learning generalizable features of deepfake audio, but rather is learning to detect specific artifacts introduced by the VITS algorithm used to generate the fake samples in the BanglaFake dataset. On the other hand, the WavLM + AASIST model, which learns directly from raw audio, may be able to learn more generalizable features of deepfake audio and thus perform better on unseen types of deepfake audio.

To test this hypothesis, we created a new dataset of TTS-generated fake audio samples using two different TTS models (Gemini and Crikk) that were not used in the original BanglaFake dataset. We then evaluated our LSTM model on this new dataset to see if it could generalize to these new types of deepfake audio.

# 3. TTS Generated Fake Audio Dataset :

- Statistics:
  - Total samples: 1921
  - Gemini TTS generated samples: 921
  - Crikk TTS generated samples: 1000

## Augmented Evaluation Dataset

- Statistics:
  - Total samples: 7321
  - Real audio samples: 3400
    - LSTM Bangla Audio Dataset: 1000
    - BanglaFake Dataset: 2400
  - Deepfake audio samples: 3921
    - LSTM Bangla Audio Dataset: 1000
    - BanglaFake Dataset: 1000
    - TTS Generated Fake Audio Dataset: 921 (Gemini) + 1000 (Crikk)

### Comparison between BanglaFake and Final Augmented Dataset

1. Dataset Distribution

![BanglaFake](./images/bangla-fake/dataset_distribution.png)
![Final Augmented Dataset](./images/aug-dataset/dataset_distribution.png)

2. Audio Statistics

![BanglaFake](./images/bangla-fake/audio_statistics.png)
![Final Augmented Dataset](./images/aug-dataset/audio_statistics.png)

3. MFCC Analysis

![BanglaFake](./images/bangla-fake/mfcc_analysis.png)
![Final Augmented Dataset](./images/aug-dataset/mfcc_analysis.png)

4. Spectrogram Comparison

![BanglaFake](./images/bangla-fake/spectrogram_comparison.png)
![Final Augmented Dataset](./images/aug-dataset/spectrogram_comparison.png)

5. Waveform Comparison

![BanglaFake](./images/bangla-fake/waveform_comparison.png)
![Final Augmented Dataset](./images/aug-dataset/waveform_comparison.png)

6. High Frequency Analysis

![BanglaFake](./images/bangla-fake/high_frequency_analysis.png)
![Final Augmented Dataset](./images/aug-dataset/high_frequency_analysis.png)

7. t-SNE MFCC Visualization

![BanglaFake](./images/bangla-fake/tsne_mfcc.png)
![Final Augmented Dataset](./images/aug-dataset/tsne_mfcc.png)

8. Source Label Distribution

![BanglaFake](./images/bangla-fake/source_label_distribution.png)
![Final Augmented Dataset](./images/aug-dataset/source_label_distribution.png)

9. Split Distribution

![BanglaFake](./images/bangla-fake/split_distribution.png)
![Final Augmented Dataset](./images/aug-dataset/split_distribution.png)

10. Multiclass Spectrogram Comparison (Final Augmented Dataset only)

![Final Augmented Dataset](./images/aug-dataset/spectrogram_multiclass_comparison.png)

11. Multiclass Waveform Comparison (Final Augmented Dataset only)

![Final Augmented Dataset](./images/aug-dataset/waveform_multiclass_comparison.png)
