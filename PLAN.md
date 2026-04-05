# **Advancing Audio Forensics in Low-Resource Languages: A Comprehensive Framework for Bengali Audio Deepfake Detection**

c

## **2\. Review of State-of-the-Art Detection Methodologies**

The field of audio deepfake detection has witnessed a rapid evolution, moving from handcrafted signal processing features to end-to-end deep learning. This section analyzes the progression of these methods and their applicability to the Bengali context.

### **2.1 Handcrafted Acoustic Features**

Historically, detection relied on feature engineering—extracting specific properties of audio known to be difficult for synthesizers to reproduce.

| Feature Type | Description | Utility in Deepfake Detection | Limitation for VITS/Modern TTS |
| :---- | :---- | :---- | :---- |
| **MFCC** (Mel-Frequency Cepstral Coefficients) | Represents the short-term power spectrum of sound on a non-linear mel scale. | Standard for speech recognition; captures timbre. | Compresses high-frequency details where vocoder artifacts often reside. **BanglaFake** analysis shows high overlap between real/fake MFCCs.4 |
| **LFCC** (Linear Frequency Cepstral Coefficients) | Similar to MFCC but uses a linear filter bank. | Better at capturing high-frequency artifacts than MFCC. | Still relies on the magnitude spectrum, ignoring phase information which is crucial for detecting neural vocoder artifacts. |
| **CQCC** (Constant Q Cepstral Coefficients) | Uses geometrically spaced frequency bins. | Excellent for analyzing pitch and tonal manipulation. | Computationally intensive; performance degrades on raw waveform synthesis which mimics natural pitch well. |

While these features provided a strong baseline for early concatenation-based fakes, the **BanglaFake** study demonstrates that for VITS-generated audio, the feature space of MFCCs for real and fake audio is largely non-separable linearly.8 This necessitates the move to learned representations.

### **2.2 Deep Learning Architectures**

The current state-of-the-art (SOTA) is dominated by deep neural networks that learn features directly from data.

#### **2.2.1 Recurrent Neural Networks (RNN/LSTM)**

Previous research on Bengali deepfake detection, specifically the work by Dipto et al. 5, explored the use of Long Short-Term Memory (LSTM) networks.

* **Methodology:** They utilized MFCCs as inputs to an LSTM to capture temporal dependencies.  
* **Performance:** Achieved \~89% accuracy on a custom dataset.  
* **Critique:** LSTMs process data sequentially, which makes them slow to train and prone to forgetting long-range dependencies in high-sample-rate audio. Furthermore, relying on MFCCs as input inherits the limitations of that feature set. While a solid baseline, LSTMs have largely been superseded by parallelizable architectures like Transformers and Convolutional networks in the broader domain.

#### **2.2.2 End-to-End Convolutional Models: RawNet2**

**RawNet2** 9 represents a paradigm shift to processing raw audio waveforms, bypassing standard feature extraction entirely.

* **Architecture:** It employs **Sinc-convolution layers** as the first layer. Instead of learning arbitrary kernels, these filters are constrained to the shape of band-pass filters (sinc functions). The network learns only the cutoff frequencies.  
* **Relevance:** This allows the model to automatically tune into specific frequency bands that contain discriminative artifacts (e.g., the specific 8kHz+ bands where HiFi-GAN creates checkerboard artifacts).  
* **Status:** RawNet2 is a strong baseline and is widely used in ASVspoof challenges. It captures fine-grained temporal cues that spectral models might smooth over.

#### **2.2.3 Graph Attention Networks: AASIST**

**AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** 9 is currently considered one of the most effective architectures for this task.

* **Concept:** AASIST models the audio signal as a graph. It extracts two types of nodes:  
  1. **Spectral Nodes:** Representing information in the frequency domain.  
  2. **Temporal Nodes:** Representing information in the time domain.  
* **Mechanism:** A **Heterogeneous Stacking Graph Attention Layer (HS-GAL)** allows information to flow between these domains. The model can learn complex relationships, such as "a specific frequency anomaly that occurs only at specific rhythmic intervals."  
* **Performance:** AASIST consistently outperforms ResNet and RawNet2 baselines on the ASVspoof benchmarks, reducing the Equal Error Rate (EER) significantly.12 Its ability to integrate global context makes it particularly robust against high-quality TTS like VITS.

#### **2.2.4 Self-Supervised Learning (SSL): Wav2Vec 2.0 & XLSR**

The most significant recent advancement is the use of **Self-Supervised Learning (SSL)** models like **Wav2Vec 2.0**.13

* **Pre-training:** These models are pre-trained on thousands of hours of unlabeled audio to solve a contrastive task (predicting masked latent representations). This forces the model to learn a rich, contextual understanding of speech structure, phonemes, and prosody.  
* **XLSR (Cross-Lingual Speech Representation):** The **XLSR-53** variant is pre-trained on 53 languages, including Bengali.15  
* **Application:** By using a pre-trained XLSR model as a front-end feature extractor, a deepfake detector can leverage the model's "knowledge" of what natural Bengali speech *should* sound like. Deviations from this learned prior (even if acoustically subtle) generate distinct embeddings that a downstream classifier can easily separate. This effectively solves the low-resource data problem by transferring knowledge from massive global datasets.

### **2.3 Synthesis of Methods for Bengali**

The optimal approach for Bengali, therefore, is not to rely on a single technique but to hybridize them. Combining the linguistic robustness of **Wav2Vec 2.0 XLSR** (to handle Bengali phonetics) with the artifact-sensitivity of **AASIST** or **RawNet2** (to catch VITS generation errors) offers the highest probability of success.

## **3\. Dataset Analysis and Expansion Strategy**

The cornerstone of this research is the **BanglaFake** dataset.4 However, to ensure the model is **algorithm agnostic** and does not overfit to VITS, we must augment this dataset with diverse synthesis sources.

### **3.1 The Core Corpus: BanglaFake**

The dataset is constructed to be balanced and statistically significant, addressing the data scarcity issue inherent to low-resource languages.

| Attribute | Details |
| :---- | :---- |
| **Total Samples** | **25,520** utterances |
| **Real Audio Count** | 12,260 (Source: SUST TTS Corpus & Mozilla Common Voice) |
| **Fake Audio Count** | 13,260 (Source: VITS trained on SUST) |
| **Sampling Rate** | 22,050 Hz |

**Limitation:** The "Fake" subset is mono-algorithmic (VITS only). A detector trained solely on this may learn to detect *VITS* rather than *Deepfakes*.

### **3.2 Dataset Expansion: The "Bengali-Wild" Augmentation**

To mitigate algorithmic bias, we will generate an additional **3,000 \- 5,000 synthetic samples** using varied Text-to-Speech architectures. This creates a "Multi-Source" training environment.

#### **3.2.1 Source A: Crikk (Proprietary/Black-Box)**

We will leverage **Crikk's Bengali TTS service** to generate high-quality synthetic audio.

* **Rationale:** Crikk utilizes modern, proprietary AI voices that likely differ in architecture (possibly WaveNet or Tacotron-based) from the open-source VITS model.  
* **Implementation:** We will use the Crikk API or web interface to synthesize Bengali sentences from the Mozilla Common Voice transcriptions. This ensures the *content* is consistent with the real dataset, but the *acoustic signature* is different.  
* **Target:** \~1,500 samples of mixed gender (Male/Female) to balance the male-dominated SUST corpus.

#### **3.2.2 Source B: Transformer-Based TTS (Orpheus)**

We will utilize the **Orpheus-Bangla-TTS** models available on Hugging Face (e.g., asif00/orpheus-bangla-tts).

* **Rationale:** Orpheus is a **Transformer-based** architecture (similar to LLaMA for audio). Its artifact distribution (attention-based discontinuities) is fundamentally different from the convolution-based artifacts of VITS.  
* **Benefit:** Training on both VITS (CNN/Flow) and Orpheus (Transformer) forces the detector to learn generalized features of synthesis rather than architecture-specific glitches.

#### **3.2.3 Source C: Emotion-Controlled TTS**

We will include samples from **Emotional TTS models** (e.g., ehzawad/orpheus-bangla-emotional-tts).

* **Rationale:** Deepfakes often struggle with emotional consistency. Including synthetic emotional speech prevents the model from simply flagging "flat prosody" as fake, pushing it to find deeper acoustic evidence of forgery.

#### **3.2.4 Source D: Gemini Pro (Google's TTS)**

* **Rationale:** Google's TTS models are among the most advanced and widely used. If we can access Bengali synthesis from Gemini Pro, it would provide a critical test of the model's generalization to commercial-grade deepfakes.
We generated `1000` samples from Gemini Pro's Bengali TTS to include in the training set across different speakers and emotional tones.

### **3.3 Preliminary Data Analysis Findings**

An initial exploratory data analysis (EDA) of the core BanglaFake dataset has yielded critical insights that shape our downstream architecture and augmentation strategies:

1.  **The "Smoking Gun" (t-SNE of MFCCs):**
    *   **Finding:** t-SNE visualization of MFCC features shows significant overlap between Real and Fake clusters.
    *   **Implication:** Linear or shallow classifiers using standard acoustic features will fail. This empirically validates the requirement for deep, non-linear representations (XLSR-AASIST).
2.  **The "Normalization Trap" (Amplitude/RMS):**
    *   **Finding:** Real audio exhibits a wide dynamic range, whereas VITS-generated fake audio is consistently normalized (compressed peak amplitude).
    *   **Implication:** There is a high risk the model will lazily learn that "consistent volume = fake." We must counter this with aggressive amplitude augmentation.
3.  **Vocoder Artifacts (High-Frequency):**
    *   **Finding:** Real audio consistently retains higher energy ratios in frequencies >8kHz compared to the VITS samples (0.138 vs 0.124 mean ratio).
    *   **Implication:** This confirms that the VITS vocoder (HiFi-GAN) struggles with high-frequency harmonics. The model must be trained at a sampling rate of at least 22,050Hz to capture these cues.

## **4. Proposed Architecture: The "Bengali-Vigil" System**

To achieve a detector that is both **robust** (to channel noise) and **algorithm agnostic** (capable of detecting unseen deepfake methods beyond VITS), we propose the **XLSR-AASIST-OC** architecture. This system enhances the standard hybrid approach with explicit generalization techniques: **RawBoost** for data augmentation and **One-Class Learning** for the loss objective.

### **4.1 High-Level Architecture Design**

The system operates in three stages:

1. **Front-End (Feature Extraction):** **Wav2Vec 2.0 XLSR-53** (Self-Supervised Learning).  
2. **Back-End (Pattern Recognition):** **AASIST** (Graph Attention Network).  
3. **Generalization Head:** **One-Class Softmax (OC-Softmax)** Classifier.

$$\\text{Audio } (x) \\xrightarrow{\\text{XLSR}} \\text{Embeddings } (E) \\xrightarrow{\\text{AASIST}} \\text{Graph Features } (G) \\xrightarrow{\\text{OC-Softmax}} \\text{Score } (S)$$

### **4.2 Detailed Component Specification**

#### **4.2.1 Front-End: Wav2Vec 2.0 XLSR-53**

Instead of using raw waveforms or MFCCs, we utilize the **wav2vec2-large-xlsr-53** model.15

* **Algorithm Agnosticism:** SSL models like XLSR learn general speech representations rather than overfitting to specific artifacts (like the checkerboard patterns of a specific GAN). By using XLSR, the model focuses on "what makes speech human" rather than "what makes VITS fake."17  
* **Configuration:** We employ a **Partial Fine-Tuning** strategy. We freeze the convolutional feature encoder and the first 12 layers of the Transformer. We fine-tune only the top 12 layers on the BanglaFake dataset. This retains the model's broad linguistic knowledge while adapting to the detection task.18

#### **4.2.2 Back-End: AASIST with RawBoost**

The embeddings from the XLSR front-end serve as the input nodes for the AASIST network.9

* **RawBoost Data Augmentation:** To prevent the model from learning simple channel cues (e.g., "VITS audio has no background noise"), we apply **RawBoost** during training. RawBoost algorithmically injects:  
  * **Linear and Non-Linear Distortion:** Simulating signal clipping and amplifier non-linearities.  
  * **Impulsive Noise:** Simulating real-world recording glitches.  
  * **Colored Noise:** Adding stationary noise to mask simple spectral artifacts.  
  * **Amplitude/Gain Perturbation:** (Crucial) Randomly scaling the signal gain to prevent the model from overfitting to the normalized volume of deepfakes.
    This forces the AASIST network to learn robust, intrinsic artifacts of synthesis rather than extrinsic channel traits.

#### **4.2.3 Classification Head: One-Class Softmax (OC-Softmax)**

Standard binary classification (Real vs. Fake) often fails on unseen attacks because it tries to learn a boundary between "Real" and "VITS." If a new attack (e.g., from Crikk) appears, it might fall on the "Real" side of that boundary.

* **Solution:** We implement **OC-Softmax** (One-Class Softmax).  
* **Mechanism:** Instead of separating two classes, OC-Softmax trains the model to compact all "Real" samples into a tight cluster in the embedding space. The "Fake" samples (from VITS, Crikk, and Orpheus) are pushed away from this cluster with a margin.  
* **Benefit:** During inference, any audio that does not fall inside the tight "Real" cluster is rejected. This makes the system **algorithm agnostic** because it doesn't need to know what the fake looks like; it only needs to know that it *doesn't* look like the diverse set of real Bengali speech it learned.

### **4.3 Training Strategy**

* **Data Split:** A **Stratified 70/15/15 Split** (Training: \~17,864, Validation: \~3,828, Testing: \~3,828).  
  * *Crucial:* Ensure the **Test Set** contains samples from the "Bengali-Wild" dataset (Crikk/Orpheus) that were *not* seen during training to rigorously test generalization.  
* Loss Function: OC-Softmax Loss.

  $$L\_{OC} \= \- \\log \\frac{e^{\\alpha(w\_y^T x\_i)}}{e^{\\alpha(w\_y^T x\_i)} \+ \\sum\_{j \\neq y} e^{\\alpha(w\_j^T x\_i)}}$$

  Where genuine samples are forced towards a center $w\_y$ and spoof samples are marginalized away.  
* **Optimizer:** **AdamW** with differential learning rates ($10^{-4}$ for AASIST, $10^{-5}$ for XLSR).

## **5\. Evaluation Metrics and Benchmarking**

To rigorously validate the "Bengali-Vigil" system, we define a comprehensive suite of metrics. These metrics are chosen to align with international standards (e.g., ASVspoof challenges) while addressing the specific needs of the application.

### **5.1 Primary Metrics**

#### **5.1.1 Equal Error Rate (EER)**

The EER is the de facto standard metric for biometric security and deepfake detection.5

* **Definition:** The specific threshold value on the ROC (Receiver Operating Characteristic) curve where the **False Acceptance Rate (FAR)** is equal to the **False Rejection Rate (FRR)**.  
  * *FAR:* The percentage of Fake audio incorrectly classified as Real.  
  * *FRR:* The percentage of Real audio incorrectly classified as Fake.  
* **Target:** A lower EER indicates a better system. The baseline LSTM model on a similar task achieved \~11% error (89% accuracy). Our target for the XLSR-AASIST architecture is an **EER \< 5%**, approaching the \<1% standards seen in English benchmarks.12

#### **5.1.2 min-tDCF (Minimum Tandem Detection Cost Function)**

While EER assumes equal cost for false alarms and misses, real-world systems often have different priorities.

* **Definition:** A metric that weights FAR and FRR differently based on the application's "cost." For a security system, admitting a deepfake (False Acceptance) is often more costly than rejecting a real user (False Rejection).  
* **Relevance:** This metric is standard in the ASVspoof challenge 11 and provides a more nuanced view of the detector's practical utility.

### **5.2 Secondary Classification Metrics**

* **Accuracy:** The overall percentage of correct predictions. (Baseline to beat: 91% from WaveNet study 5).  
* **Precision & Recall:**  
  * *Precision:* Of all audio flagged as fake, how many were actually fake? (High precision \= low false alarms).  
  * *Recall:* Of all actual fakes, how many did we catch? (High recall \= high security).  
* **F1-Score:** The harmonic mean of Precision and Recall.

### **5.3 Visualization and Explainability**

Quantitative metrics must be supported by qualitative analysis to ensure the model is learning the *right* features.

* **t-SNE of Embeddings:** We will plot the t-SNE of the *final layer embeddings* (before the Softmax). Unlike the raw MFCC t-SNE shown in the BanglaFake paper 4, which showed overlap, the learned embeddings from XLSR-AASIST should show clear, distinct clusters for "Real" and "Fake" audio.  
* **Attention Maps:** By visualizing the attention weights of the AASIST layers, we can generate heatmaps over the spectrograms.  
  * *Hypothesis (Confirmed):* Data analysis shows real audio has higher energy ratios >8kHz. The model should attend heavily to these high-frequency regions and silence transitions, where VITS artifacts are most prominent.20 If the model attends only to the fundamental frequency ($F_0$), it might be overfitting to the speaker's pitch rather than detecting artifacts.

## **6\. Challenges, Limitations, and Future Outlook**

### **6.1 The Generalization Gap**

A critical risk is that the model trained on VITS fakes will fail to detect fakes generated by other architectures (e.g., Diffusion models like DiffWave or older autoregressive models like Tacotron). This is known as the "unseen attack" problem. The graph-based nature of AASIST helps, but true robustness requires a diverse training set. Future iterations should augment BanglaFake with samples generated by other TTS engines.

### **6.2 The "Arms Race"**

As detection models become public, attackers will use them as discriminators to train better generators (Adversarial Training). This creates a perpetual cycle. To stay ahead, detection research must pivot towards **generalized artifact detection**—finding fundamental inconsistencies in neural audio generation (e.g., lack of breath sounds, unnatural phase coherence) rather than overfitting to specific model signatures.

### **6.3 Codec and Channel Robustness**

Real-world audio is rarely lossless WAV. It is compressed via WhatsApp, Zoom, or cellular networks. Compression algorithms (like AAC or GSM) act as low-pass filters, potentially removing the high-frequency artifacts that detectors rely on.21 The proposed training augmentation (adding compression noise) is a mitigation strategy, but evaluating the model on "transcoded" test sets is essential for deployment readiness.

## **7\. Conclusion**

The creation of a Bengali Audio Deepfake Detector is not merely a technical challenge but a societal imperative. The **BanglaFake** dataset provides the necessary fuel—25,000+ samples of linguistically rich, VITS-generated audio—to drive this research. However, the sophistication of VITS means that traditional spectral analysis is insufficient.

This report proposes the **XLSR-AASIST** architecture, a system that marries the best of Self-Supervised Learning with Graph Neural Networks. By fine-tuning a multilingual model (XLSR) to the nuances of Bengali phonology and employing a graph network (AASIST) to scrutinize spectro-temporal consistency, we can construct a defense mechanism that is both accurate and robust. With rigorous evaluation using EER and min-tDCF, and a clear eye on the challenges of generalization, this framework paves the way for trusted digital audio in the Bengali-speaking world.

### ---

**Appendix: Implementation Roadmap**

| Phase | Task | Tools/Libraries | Estimated Timeline |
| :---- | :---- | :---- | :---- |
| **1\. Data Prep** | Download BanglaFake, verify integrity, generate stratified splits (70/15/15). | HuggingFace Datasets, Pandas, Librosa | Week 1 |
| **2\. Augmentation** | Generate 3k samples using **Crikk** and **Orpheus-Bangla** models. | Crikk API, HuggingFace Inference | Week 2 |
| **3\. Baseline** | Train RawNet2 on the expanded dataset. | PyTorch, Torchaudio | Week 3 |
| **4\. Front-End** | Implement and fine-tune Wav2Vec 2.0 XLSR-53. Freeze lower layers. | HuggingFace Transformers | Week 4-5 |
| **5\. Back-End** | Implement AASIST graph modules. Connect with XLSR embeddings. | PyTorch Geometric | Week 6-7 |
| **6\. Training** | Full system training with augmentations (Noise, RIR, MP3). | PyTorch Lightning, NVIDIA GPU | Week 8-9 |
| **7\. Eval** | Compute EER, min-tDCF. Generate t-SNE plots and attention maps. | Scikit-learn, Matplotlib | Week 10 |

**Recommended Codebase References:**

* **BanglaFake Generation:**(https://github.com/KamruzzamanAsif/BanglaFake) 22  
* **AASIST Implementation:** [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist) 9  
* **Wav2Vec 2.0 Fine-tuning:** HuggingFace Wav2Vec2ForSequenceClassification 13

#### **Works cited**

1. Understanding VITS: Revolutionizing Voice AI With Natural-Sounding Speech \- Vapi AI Blog, accessed December 11, 2025, [https://vapi.ai/blog/vits](https://vapi.ai/blog/vits)  
2. An Emotion Speech Synthesis Method Based on VITS \- MDPI, accessed December 11, 2025, [https://www.mdpi.com/2076-3417/13/4/2225](https://www.mdpi.com/2076-3417/13/4/2225)  
3. HiFi-GAN: Real-Time AI Audio Generation Explained. \- Dialora.ai, accessed December 11, 2025, [https://www.dialora.ai/blog/hifi-gan-ai-audio-generation-guide](https://www.dialora.ai/blog/hifi-gan-ai-audio-generation-guide)  
4. BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset, accessed December 11, 2025, [https://arxiv.org/html/2505.10885v1](https://arxiv.org/html/2505.10885v1)  
5. Detecting Bangla DeepFake Audio: A Dual Approach Using LSTM and WaveNet | Request PDF \- ResearchGate, accessed December 11, 2025, [https://www.researchgate.net/publication/396986122\_Detecting\_Bangla\_DeepFake\_Audio\_A\_Dual\_Approach\_Using\_LSTM\_and\_WaveNet](https://www.researchgate.net/publication/396986122_Detecting_Bangla_DeepFake_Audio_A_Dual_Approach_Using_LSTM_and_WaveNet)  
6. SUST TTS Corpus: A phonetically-balanced corpus for Bangla text-to-speech synthesis, accessed December 11, 2025, [https://www.researchgate.net/publication/355823823\_SUST\_TTS\_Corpus\_A\_phonetically-balanced\_corpus\_for\_Bangla\_text-to-speech\_synthesis](https://www.researchgate.net/publication/355823823_SUST_TTS_Corpus_A_phonetically-balanced_corpus_for_Bangla_text-to-speech_synthesis)  
7. BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset \- arXiv, accessed December 11, 2025, [https://arxiv.org/pdf/2505.10885](https://arxiv.org/pdf/2505.10885)  
8. BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset, accessed December 11, 2025, [https://www.researchgate.net/publication/391856638\_BanglaFake\_Constructing\_and\_Evaluating\_a\_Specialized\_Bengali\_Deepfake\_Audio\_Dataset](https://www.researchgate.net/publication/391856638_BanglaFake_Constructing_and_Evaluating_a_Specialized_Bengali_Deepfake_Audio_Dataset)  
9. aasist: audio anti-spoofing using integrated spectro-temporal graph attention networks \- Eurecom, accessed December 11, 2025, [https://www.eurecom.fr/publication/6696/download/sec-publi-6696.pdf](https://www.eurecom.fr/publication/6696/download/sec-publi-6696.pdf)  
10. Advanced RawNet2 with Attention-based Channel Masking for Synthetic Speech Detection \- ISCA Archive, accessed December 11, 2025, [https://www.isca-archive.org/interspeech\_2023/li23h\_interspeech.pdf](https://www.isca-archive.org/interspeech_2023/li23h_interspeech.pdf)  
11. (PDF) AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks \- ResearchGate, accessed December 11, 2025, [https://www.researchgate.net/publication/355060497\_AASIST\_Audio\_Anti-Spoofing\_using\_Integrated\_Spectro-Temporal\_Graph\_Attention\_Networks](https://www.researchgate.net/publication/355060497_AASIST_Audio_Anti-Spoofing_using_Integrated_Spectro-Temporal_Graph_Attention_Networks)  
12. Replay Attacks Against Audio Deepfake Detection \- arXiv, accessed December 11, 2025, [https://arxiv.org/html/2505.14862v2](https://arxiv.org/html/2505.14862v2)  
13. Wav2Vec2 \- Hugging Face, accessed December 11, 2025, [https://huggingface.co/docs/transformers/model\_doc/wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)  
14. Wav2DF-TSL: Two-stage Learning with Efficient Pre-training and Hierarchical Experts Fusion for Robust Audio Deepfake Detection \- arXiv, accessed December 11, 2025, [https://arxiv.org/html/2509.04161v1](https://arxiv.org/html/2509.04161v1)  
15. Wav2vec2 Large Xlsr Bengali · Models \- Dataloop, accessed December 11, 2025, [https://dataloop.ai/library/model/tanmoyio\_wav2vec2-large-xlsr-bengali/](https://dataloop.ai/library/model/tanmoyio_wav2vec2-large-xlsr-bengali/)  
16. arijitx/wav2vec2-large-xlsr-bengali \- Hugging Face, accessed December 11, 2025, [https://huggingface.co/arijitx/wav2vec2-large-xlsr-bengali](https://huggingface.co/arijitx/wav2vec2-large-xlsr-bengali)  
17. Related Work \- Deepfake Total, accessed December 11, 2025, [https://deepfake-total.com/related\_work/](https://deepfake-total.com/related_work/)  
18. Practical Guide on Fine-Tuning Wav2Vec2 | by Hey Amit \- Medium, accessed December 11, 2025, [https://medium.com/@heyamit10/practical-guide-on-fine-tuning-wav2vec2-7c343d5d7f3b](https://medium.com/@heyamit10/practical-guide-on-fine-tuning-wav2vec2-7c343d5d7f3b)  
19. Low-rank Adaptation Method for Wav2vec2-based Fake Audio Detection \- arXiv, accessed December 11, 2025, [https://arxiv.org/pdf/2306.05617](https://arxiv.org/pdf/2306.05617)  
20. AI-Synthesized Voice Detection Using Neural Vocoder Artifacts \- CVF Open Access, accessed December 11, 2025, [https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Sun\_AI-Synthesized\_Voice\_Detection\_Using\_Neural\_Vocoder\_Artifacts\_CVPRW\_2023\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.pdf)  
21. Replay Attacks Against Audio Deepfake Detection \- ISCA Archive, accessed December 11, 2025, [https://www.isca-archive.org/interspeech\_2025/muller25\_interspeech.pdf](https://www.isca-archive.org/interspeech_2025/muller25_interspeech.pdf)  
22. BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset \- GitHub, accessed December 11, 2025, [https://github.com/kamruzzamanasif/banglafake](https://github.com/kamruzzamanasif/banglafake)