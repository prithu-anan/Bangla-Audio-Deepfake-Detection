# Deep Research Prompt for Reference Hunting

You are an expert research assistant specializing in speech processing, audio deepfake detection, speech forensics, self-supervised learning, and low-resource language technologies.

I am preparing an IEEE SLT paper on Bangla Audio Deepfake Detection. The latex code of the paper is attached for better understanding.

Your task is NOT simply to find papers matching keywords.

Instead, for each scientific claim listed below:

1. Find the strongest peer-reviewed evidence supporting the claim.
2. Prioritize IEEE, ACM, Interspeech, ICASSP, ASRU, SLT, TASLP, Computer Speech and Language, Pattern Recognition, Expert Systems with Applications, Engineering Reports, and highly cited arXiv papers.
3. Prefer survey papers when a broad claim is made.
4. Prefer benchmark papers when discussing generalization or robustness.
5. Prefer original papers when discussing specific methods (WavLM, HuBERT, wav2vec 2.0, AASIST, RawBoost).
6. For each claim return:

   * Claim text
   * Recommended citation(s) in latex style
   * Why the citation supports the claim
   * BibTeX entries to be added references.bib
   * Confidence score (1–5)

Claims requiring references:

[Find the claims here](#claims-that-must-have-references)

Special Instructions:

If no paper directly supports a claim, identify the nearest defensible scientific statement and find citations supporting that statement instead.

Do not fabricate references.

Return only papers that actually exist.

Prioritize references published between 2020 and 2026, except for foundational statistical references (e.g., Cohen's effect size work).

One additional reviewer observation: the paper actually needs **more references in the Introduction** than in Related Work. IEEE SLT reviewers often judge novelty from the Introduction, and currently several foundational claims there are uncited. If you add 8–12 strong references to the Introduction alone, the manuscript will look substantially more mature.



# Claims That MUST Have References

These are statements that make general scientific claims and should almost certainly be supported by citations.

---

## 1. Introduction

### Claim A

> "Recent advances in neural text-to-speech (TTS) and voice conversion systems have enabled the generation of highly realistic synthetic speech, creating growing concerns for misinformation, impersonation, and voice-based fraud."

Need citations for:

1. Modern neural TTS realism
2. Voice conversion realism
3. Fraud / misinformation risks

Recommended reference types:

* Audio deepfake survey papers
* Audio deepfake threat assessment papers
* Voice phishing / vishing studies

---

### Claim B

> "While audio deepfake detection has received increasing attention, most existing research focuses on high-resource languages and source-constrained evaluation settings that may not reflect real-world deployment conditions."

Need references supporting:

* Dominance of English/high-resource language datasets
* Source-constrained evaluation protocols
* Lack of real-world generalization

---

### Claim C

> "Publicly available datasets remain limited..."

Need references documenting:

* Bangla dataset scarcity
* Low-resource language challenges

---

### Claim D

> "Reported performance may reflect memorization of generator-specific artifacts rather than genuine discrimination between authentic and synthetic speech."

Very important claim.

Need:

* Generalization papers
* Cross-generator benchmark papers
* Audio deepfake robustness studies

---

### Claim E

> "The ability to generalize to previously unseen generators remains largely unexplored for Bangla audio deepfake detection."

Need:

* BanglaFake paper
* Ayan et al.
* Literature review confirming absence of cross-generator studies

---

### Claim F

> "Recent self-supervised speech models such as WavLM have demonstrated strong transferability across speech processing tasks..."

Need:

* WavLM
* HuBERT
* wav2vec 2.0

---

## 2. Related Work Section

---

### Claim G

> "Deep learning approaches based on CNNs, RNNs, and transformer architectures have become dominant..."

Need citation(s).

Prefer:

* Survey papers
* Recent review papers

---

### Claim H

> "LSTM-based models remain widely used..."

Need supporting literature.

Not just Ayan et al.

Need 2–4 papers.

---

### Claim I

> "Self-supervised speech models ... have achieved state-of-the-art performance..."

Need evidence.

Need:

* WavLM
* XLS-R
* HuBERT
* ASVspoof leaderboard style papers

---

### Claim J

> "Progress in deepfake detection has been driven by benchmark datasets such as ASVspoof and ADD..."

Need citations for:

* ASVspoof
* ADD challenge

---

### Claim K

> "These resources primarily target high-resource languages."

Need evidence.

Could be:

* Dataset language distribution analysis
* Survey papers

---

### Claim L

> "Detector robustness to unseen synthesis systems remains a major open challenge."

Very important.

Need:

* Generalization papers
* Benchmark papers

---

### Claim M

> "Many detectors learn generator-specific artifacts rather than intrinsic characteristics of synthetic speech."

Needs strong evidence.

This is one of the most important citations in the paper.

---

### Claim N

> "There is growing interest in self-supervised representations, domain-invariant learning, and source-diverse evaluation protocols."

Need references for each trend.

---

## 3. Methodology Section

Most methodology descriptions do not require citations.

However:

---

### Claim O

> "Speaker-aware and source-aware stratified splits are employed to reduce information leakage."

Need support.

Could cite:

* Dataset leakage papers
* Speaker overlap papers

---

### Claim P

> "RawBoost-inspired augmentations improve robustness under channel and generator variability."

Need citation beyond RawBoost itself.

Need papers showing augmentation improves anti-spoofing robustness.

---

### Claim Q

> "Unlike handcrafted features such as MFCCs, WavLM preserves rich temporal, spectral, and contextual information..."

Needs support.

Could cite:

* WavLM paper
* SSL speech representation papers

---

### Claim R

> "Self-supervised speech embeddings provide superior transferability and robustness..."

Needs evidence.

Should not rely only on Urdu paper.

Need broader literature.

---

### Claim S

> "AASIST models spectro-temporal dependencies using graph attention mechanisms..."

Need original AASIST citation only.

Already present.

---

### Claim T

> "Threshold-independent ranking quality is particularly important in forensic applications."

Need forensic evaluation paper.

Potential reviewer comment otherwise.

---

## 4. Evaluation Section

Most numerical findings are your own results.

No citation needed.

But:

---

### Claim U

> "This setting better reflects realistic deployment scenarios..."

Need support.

Need real-world deepfake deployment/generalization literature.

---

### Claim V

> "A robust detector should assign consistently high fake probabilities..."

Need calibration/generalization reference.

Not mandatory but helpful.

---

## 5. LOFSO Section

---

### Claim W

> "This protocol directly measures the ability of a detector to generalize to previously unseen generators."

Need reference.

Could cite:

* Leave-one-domain-out
* Cross-generator evaluation papers

---

## 6. Statistical Analysis Section

---

### Claim X

> "Large effect sizes suggest that these differences are practically meaningful rather than merely statistically significant."

Need statistics methodology reference.

Could cite:

* Cohen
* Statistical effect size literature

---

### Claim Y

> "Source effects are substantially stronger than class effects..."

No citation needed because derived from your results.

---

### Claim Z

> "Representation learning plays a central role in determining robustness under generator shift."

This is an interpretation.

Would benefit from:

* Domain generalization literature
* SSL robustness papers

---

# Claims That Reviewers Usually Ignore

These are generally accepted and don't require citations:

* Dataset descriptions
* Model architecture descriptions
* Train/validation/test split details
* Numerical results
* Discussion of your own findings


