This script is timed for approximately **8 minutes** (at a moderate speaking pace of ~130 words per minute). It is designed to accompany the LaTeX Beamer slides provided previously.

---

### **Slide 1: Title Slide (0:00 - 0:30)**
**Speaker:** "Good morning/afternoon, Professor. I am Prithu Anan, and along with my partner Zia Ul Hassan Abdullah, we are pleased to present our research on 'Bengali Audio Deepfake Detection: A Research Framework for Robust Forensic Defense.' Our work focuses on closing the gap between rapidly advancing generative models and the forensic tools available for low-resource languages like Bengali."

### **Slide 2: Outline (0:30 - 0:45)**
**Speaker:** "Today, we will walk you through the problem landscape, our comparative analysis of various datasets, the evolution of our proposed WavLM-AASIST architecture, and finally, a deep dive into our performance results and the challenges we’ve identified for future work."

### **Slide 3: The Rising Threat (0:45 - 1:30)**
**Speaker:** "The core of the problem lies in the 'Asymmetric Threat.' Modern Generative AI, specifically architectures like VITS, can now produce speech that is near-indistinguishable from humans. VITS uses a stochastic duration predictor and a HiFi-GAN decoder to remove traditional artifacts like metallic tones or spectral banding. 



Why Bengali? It is spoken by over 250 million people, yet it remains 'low-resource' in the deepfake space. Our unique phonetics—like retroflex stops and complex conjuncts—mean that generic English-trained detectors often fail to catch the subtle statistical anomalies left behind by Bengali-tuned generators."

### **Slide 4: Research Gap (1:30 - 2:00)**
**Speaker:** "Our initial investigation revealed a significant research gap. Existing models often overfit to the specific artifacts of one generator—for instance, the VITS samples in the BanglaFake dataset. When faced with 'unseen' deepfakes from modern systems like Google’s Gemini or Crikk TTS, these models fail to generalize. Our goal was to build a detector that understands the underlying manifold of Bengali speech, not just the 'noise' of one specific algorithm."

### **Slide 5: Datasets Overview (2:00 - 2:45)**
**Speaker:** "To address this, we worked with four distinct data sources. We started with the LSTM baseline dataset and the massive BanglaFake set, which contains over 26,000 samples. However, we realized BanglaFake was heavily biased toward VITS. To test real-world robustness, we created an 'Augmented Evaluation Dataset' incorporating Gemini and Crikk TTS. This allowed us to observe how models handle multiple 'fake' sources simultaneously."

### **Slides 6-11: Visual Analysis (2:45 - 3:45)**
**Speaker:** "If we look at the t-SNE visualizations of our MFCC features, the difference is clear. In the original BanglaFake set, the 'real' and 'fake' clusters are very tightly defined, making it an 'easy' task for simple models. However, in our Augmented set, the acoustic manifold is much broader. You can see in the spectrogram and high-frequency analysis that modern fakes from Gemini and Crikk hide their artifacts much more effectively in the upper frequency bands, which traditional MFCC-based models often discard."

### **Slide 12: Evolution of Models (3:45 - 4:15)**
**Speaker:** "This led to our technological shift. We moved away from the baseline LSTM model. While LSTM is computationally light, it relies on MFCCs—hand-engineered features that lose phase information and high-frequency details. Our advanced solution uses WavLM-Large paired with a Modified AASIST backend. This pipeline processes the raw waveform directly, ensuring no forensic evidence is lost during feature extraction."

### **Slide 13: WavLM + AASIST Architecture (4:15 - 5:15)**
**Speaker:** "Let’s look at the architecture. We use a fine-tuned WavLM-Large frontend, specifically focusing on the top 12 layers to capture deep speech representations. This is passed into a modified AASIST backend, which uses Graph Attention Networks. The graph nodes represent different spectral and temporal regions, allowing the model to 'connect the dots' between distant artifacts. We also integrated RawBoost augmentation to simulate real-world conditions like noise and compression before the audio even reaches the model."

### **Slide 14: Loss Function: OC-Softmax (5:15 - 5:45)**
**Speaker:** "Instead of standard Cross-Entropy, we used One-Class Softmax loss. The intuition here is to create a 'safety margin.' We want to compact all 'real' speech into a tight hypersphere in the embedding space while pushing 'fake' samples as far away as possible. This is far more effective for detection tasks where the 'fake' class is constantly evolving and unpredictable."

### **Slide 15-18: Performance & Generalization (5:45 - 6:45)**
**Speaker:** "Our results on the BanglaFake set were strong, with WavLM achieving an AUC of 0.93. But the true test was the Augmented Dataset. Here, the LSTM’s accuracy plummeted to 58%, barely better than a coin flip. In contrast, WavLM maintained a much higher AUC of 0.81. 

Looking at the confusion matrices, there is a vital trade-off. While the LSTM might have a higher raw accuracy in some cases, WavLM offers near-perfect Precision on the 'fake' class. In a forensic scenario, this is critical: when our model says a clip is fake, it is almost certainly fake, even if it is slightly more conservative in its overall recall."

### **Slide 19-21: Ablation & Holdout Results (6:45 - 7:15)**
**Speaker:** "We performed a 'Leave-One-Fake-Source-Out' ablation study. When we hid the Crikk or Gemini sources during training, WavLM showed a 21% relative improvement in AUC over the LSTM. This proves that raw audio models are learning 'general' features of synthetic speech rather than just memorizing the quirks of one specific TTS engine."

### **Slide 22-24: Challenges & Conclusion (7:15 - 8:00)**
**Speaker:** "We must conclude with the 'BanglaFake Anomaly.' Both models struggled when we held out the BanglaFake source, suggesting that older synthesis artifacts are fundamentally different from modern ones. This 'Generator Drift' is our biggest challenge. 

Moving forward, we propose using Domain-Adversarial training to make models generator-invariant, and exploring cross-lingual transfer learning from English datasets. In summary, WavLM-AASIST provides the most robust foundation for Bengali audio forensics, offering the high precision and generalization needed for real-world defense. Thank you, and we are now open for questions."