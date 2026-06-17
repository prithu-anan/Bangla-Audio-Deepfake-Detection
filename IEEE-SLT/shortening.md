You are an experienced academic editor specializing in IEEE, ACM, Elsevier, and Springer conference papers.

Your task is to reduce the length of the following paper while preserving its scientific contribution, technical correctness, and publication quality.

Requirements:

1. Preserve the core scientific message, novelty, and main contributions.
2. Do NOT remove:

   * Research motivation
   * Problem statement
   * Methodology overview
   * Experimental setup
   * Key quantitative results
   * Main discussion points
   * Conclusion and implications
3. Aggressively compress:

   * Repetitive explanations
   * Background information already familiar to experts
   * Literature review details that do not directly support the contribution
   * Redundant descriptions of datasets, models, or evaluation metrics
   * Verbose transitions and narrative text
4. Maintain academic tone and logical flow.
5. Preserve all citations and citation placement whenever possible.
6. Do not invent new claims, results, experiments, or references.
7. Prefer concise scientific writing over explanatory prose.
8. Keep all numerical results, statistical findings, and performance comparisons unless they are redundant.
9. If multiple sentences express the same idea, merge them into a single concise sentence.
10. Rewrite rather than simply delete whenever possible.

Target reduction:

* Current length: [8 pages , 7 full and one full column of the 8th page]
* Target length: [Not more than 6 pages]

Output format:

A. Summary of major reductions made
B. Revised shortened version
C. List of important content that was removed or merged

Before editing, identify:

* The paper's central research question
* Main contribution(s)
* Essential results that must not be lost

Use these as preservation constraints during compression.

Paper:

\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{subcaption}
\usepackage{booktabs}
\usetikzlibrary{arrows.meta}
\usepackage{accessibility}
\providecommand{\Description}[1]{}
\usepackage{pgfplots}
\usepackage{pgf-pie}
\usepackage{multirow}
\pgfplotsset{compat=1.18}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{Beyond Acoustic Features: Understanding Cross-Generator Generalization in Bangla Audio Deepfake Detection
% {\footnotesize \textsuperscript{*}Note: Sub-titles are not captured in Xplore and
% should not be used}
% \thanks{Identify applicable funding agency here. If none, delete this.}
}

\author{\IEEEauthorblockN{Anonymous Author(s)}
% \IEEEauthorblockA{\textit{dept. name of organization (of Aff.)} \\
% \textit{name of organization (of Aff.)}\\
% City, Country \\
% email address or ORCID}
% \and
% \IEEEauthorblockN{2\textsuperscript{nd} Given Name Surname}
% \IEEEauthorblockA{\textit{dept. name of organization (of Aff.)} \\
% \textit{name of organization (of Aff.)}\\
% City, Country \\
% email address or ORCID}
% \and
% \IEEEauthorblockN{3\textsuperscript{rd} Given Name Surname}
% \IEEEauthorblockA{\textit{dept. name of organization (of Aff.)} \\
% \textit{name of organization (of Aff.)}\\
% City, Country \\
% email address or ORCID}
% \and
% \IEEEauthorblockN{4\textsuperscript{th} Given Name Surname}
% \IEEEauthorblockA{\textit{dept. name of organization (of Aff.)} \\
% \textit{name of organization (of Aff.)}\\
% City, Country \\
% email address or ORCID}
% \and
% \IEEEauthorblockN{5\textsuperscript{th} Given Name Surname}
% \IEEEauthorblockA{\textit{dept. name of organization (of Aff.)} \\
% \textit{name of organization (of Aff.)}\\
% City, Country \\
% email address or ORCID}
% \and
% \IEEEauthorblockN{6\textsuperscript{th} Given Name Surname}
% \IEEEauthorblockA{\textit{dept. name of organization (of Aff.)} \\
% \textit{name of organization (of Aff.)}\\
% City, Country \\
% email address or ORCID}
}

\maketitle

\begin{abstract}
Audio deepfake detection remains underexplored for low-resource languages such as Bangla, where limited datasets and rapidly evolving synthesis techniques hinder the development of robust forensic systems. In this work, we augment two existing Bangla audio deepfake datasets with a newly curated collection of synthetic speech generated using modern text-to-speech systems, creating a more diverse evaluation benchmark. We propose a detector based on a WavLM frontend and an AASIST backend, and compare its performance against a representative MFCC-LSTM baseline. Through statistical analysis of handcrafted acoustic features and representation-space diagnostics, we demonstrate that traditional features are strongly influenced by generator-specific artifacts, whereas WavLM learns more generator-invariant cues for authenticity discrimination. Extensive leave-one-fake-source-out evaluations reveal substantial generator drift, with BanglaFake emerging as the most challenging unseen source. The proposed architecture achieves superior cross-generator robustness, improving mean holdout AUC by over 21\% compared to the LSTM baseline. Our findings highlight the importance of source-diverse evaluation and self-supervised representations for reliable Bangla audio deepfake detection.

\end{abstract}

\begin{IEEEkeywords}
Audio Deepfake Detection, Bangla Speech, Low-Resource Languages, Self-Supervised Learning, WavLM, AASIST, Cross-Generator Generalization, Leave-One-Fake-Source-Out (LOFSO)
\end{IEEEkeywords}

\section{Introduction}
Recent advances in neural text-to-speech (TTS) and voice conversion
systems have enabled the generation of highly realistic synthetic speech
\cite{kim2021vits, ren2021fastspeech2, wang2023valle}, creating growing
concerns for misinformation, impersonation, and voice-based
fraud \cite{shaaban2025, zhang2025ahead}. While audio deepfake detection has received increasing attention
\cite{zhang2025ahead, yi2023survey}, most existing research focuses on
high-resource languages and source-constrained evaluation settings that may
not reflect real-world deployment conditions \cite{muller2022generalize,
muller2024mlaad}.

For Bangla, the problem is particularly challenging. Publicly available datasets remain limited \cite{banglafake,
samu2025zeroshot}, and existing studies primarily evaluate detectors on a
single synthesis source \cite{bangla_lstm_wavenet, banglafake}. As a result, reported performance may reflect memorization of
generator-specific artifacts rather than genuine discrimination between
authentic and synthetic speech \cite{muller2022generalize,
chen2020generalization}. The ability to generalize to previously unseen generators remains largely
unexplored for Bangla audio deepfake detection \cite{banglafake,
bangla_lstm_wavenet, samu2025zeroshot}.

Recent self-supervised speech models such as WavLM \cite{wavlm},
HuBERT \cite{hsu2021hubert}, and wav2vec~2.0 \cite{baevski2020wav2vec2}
have demonstrated strong transferability across speech processing tasks
and offer a promising alternative to traditional handcrafted feature
pipelines. However, their effectiveness under cross-generator distribution shift in low-resource languages has not been systematically investigated. Furthermore, little is known about how handcrafted acoustic features and self-supervised representations differ in their sensitivity to generator-specific artifacts.

To address these gaps, we augment two existing Bangla deepfake datasets with a newly curated collection of synthetic speech generated using modern text-to-speech systems and develop a deepfake detection framework based on WavLM and AASIST. We evaluate the proposed system against an MFCC-LSTM baseline using both conventional testing and Leave-One-Fake-Source-Out (LOFSO) protocols. In addition, we perform statistical analyses of acoustic features and representation spaces to study generator drift and cross-generator robustness.

The main contributions of this work are:

\begin{itemize}
    \item We curate a new Bangla synthetic speech collection and augment existing Bangla deepfake datasets to create a more diverse benchmark for evaluating audio deepfake detectors.
    \item We develop and evaluate a Bangla audio deepfake detection framework based on WavLM and AASIST and compare it against a representative MFCC-LSTM baseline.
    \item We perform statistical analyses of acoustic feature distributions and learned representations to investigate the relationship between generator-specific artifacts and deepfake detection performance.
    \item We conduct comprehensive Leave-One-Fake-Source-Out evaluations to quantify cross-generator robustness and identify the most challenging synthesis sources.
    \item We provide one of the first systematic investigations of generator drift and representation learning for Bangla audio deepfake detection, contributing new insights for forensic systems in low-resource languages. 
\end{itemize}


\section{Related Work}

Early audio deepfake detection relied on handcrafted acoustic features such as MFCCs, CQCCs, and spectral descriptors combined with conventional classifiers including GMMs and SVMs \cite{yi2023survey,mfccml}. More recently, deep learning approaches based on CNNs, RNNs, and transformer architectures have become dominant, enabling end-to-end learning of spoofing artifacts from speech signals \cite{zhang2025ahead,shaaban2025}. Among these approaches, LSTM-based models remain widely used due to their simplicity and effectiveness on source-constrained datasets \cite{bangla_lstm_wavenet,shaaban2025}, while self-supervised speech models such as wav2vec~2.0
\cite{baevski2020wav2vec2}, HuBERT \cite{hsu2021hubert}, and WavLM
\cite{wavlm} have achieved state-of-the-art performance by learning
robust speech representations directly from raw audio
\cite{tak2022sslantispoofing, wang2022sslfrontends}. Architectures such as AASIST further improve detection by modeling spectro-temporal dependencies using graph-attention mechanisms \cite{aasist}.

Progress in deepfake detection has been driven by benchmark datasets such
as ASVspoof \cite{wang2020asvspoof2019, liu2023asvspoof2021} and ADD
\cite{yi2022add}; however, these resources primarily target high-resource
languages \cite{yi2023survey, muller2024mlaad}. For Bengali, the recently introduced BanglaFake dataset provides over 25,000 real and synthetic utterances generated using a VITS-based pipeline \cite{kim2021vits} and highlights the substantial overlap between real and fake samples in MFCC space \cite{banglafake}. Similar challenges have been reported for other low-resource languages. For example, Owais \emph{et al.} demonstrated that transformer-based architectures outperform CNN and CNN-LSTM models for Urdu deepfake detection under augmentation and perturbation settings \cite{urdu2026}.

Despite strong benchmark performance, detector robustness to unseen synthesis systems remains a major open challenge. Recent surveys and benchmark studies consistently report significant performance degradation under cross-generator evaluation \cite{yi2023survey,zhang2025ahead,review2024,muller2022generalize,muller2024harder}. Studies such as ``Does Audio Deepfake Detection Generalize?''
\cite{muller2022generalize} and subsequent analysis \cite{muller2024harder,
chen2020generalization} show that many detectors learn generator-specific
artifacts rather than intrinsic characteristics of synthetic speech. Consequently, there is growing interest in self-supervised representations
\cite{tak2022sslantispoofing, wang2022sslfrontends}, domain-invariant
learning, and source-diverse evaluation protocols \cite{muller2024mlaad,
muller2022generalize}.

Existing Bengali deepfake detection studies primarily evaluate models on data generated by a single synthesis pipeline \cite{bangla_lstm_wavenet,banglafake}, leaving the robustness of current detectors to unseen Bangla speech generators largely unexplored. In this work, we address this gap through multi-source Bangla deepfake evaluation using synthetic speech generated from multiple modern TTS systems, Leave-One-Fake-Source-Out (LOFSO) testing, and complementary statistical analyses of acoustic and representation-level generator drift.


\section{Methodology}

\subsection{Problem Definition}

The objective of this work is to perform binary audio deepfake detection. Given an input speech recording $x$, the detector predicts whether the utterance is authentic human speech or AI-generated synthetic speech.

In this study, \emph{real} (or \emph{bona fide}) audio refers to speech naturally produced by human speakers and recorded through conventional acquisition processes without the use of speech synthesis or voice conversion technologies. Conversely, \emph{fake}, \emph{synthetic}, \emph{deepfake}, and \emph{spoofed} audio refer to speech generated wholly or predominantly by neural speech generation systems, including text-to-speech (TTS) models.

Formally, the task is defined as learning a binary classifier

\begin{equation}
f(x) \rightarrow y,
\end{equation}

where

\begin{equation}
y \in {\text{real}, \text{fake}}.
\end{equation}

The primary challenge addressed in this work is not merely distinguishing real and synthetic speech under source-matched conditions, but generalizing to synthetic speech generated by previously unseen TTS systems.


\subsection{Datasets}
% review comment
To evaluate Bangla audio deepfake detection under both source-matched and source-shifted conditions, we utilize two publicly available Bengali datasets and two newly curated synthetic speech collections. In source-matched evaluation, training and test data contain speech generated by the same synthesis sources, whereas source-shifted evaluation assesses generalization to generators unseen during training. Table~\ref{tab:datasets} summarizes the data sources.

\begin{table}[t]
\centering
\caption{Datasets used in this study.}
\label{tab:datasets}
\begin{tabular}{lccc}
\hline
Dataset & Total & Real & Fake \\
\hline
BanglaFake \cite{banglafake} & 26,592 & 13,796 & 12,796 \\
LSTM Bangla Dataset \cite{bangla_lstm_wavenet} & 4,500 & 2,250 & 2,250 \\
Crikk Deepfake Dataset & 1000 & 0 & 1000 \\
Gemini Deepfake Dataset & 911 & 0 & 911 \\
\hline
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Composition of the final evaluation benchmark.}
\label{tab:benchmark_composition}
\begin{tabular}{lcc}
\hline
Class & Source & Samples \\
\hline
banglafake\_real & BanglaFake & 2400 \\
lstm\_real & LSTM Dataset & 1000 \\
banglafake\_deepfake & BanglaFake & 1000 \\
crikk\_deepfake & Crikk TTS & 1000 \\
gemini\_deepfake & Gemini TTS & 911 \\
lstm\_deepfake & LSTM Dataset & 1000 \\
\hline
Total & --- & 7311 \\
\hline
\end{tabular}
\end{table}

The primary training corpus is the BanglaFake dataset, one of the first large-scale Bengali deepfake audio datasets, containing both bona fide and synthetic utterances generated using a VITS-based speech synthesis pipeline \cite{banglafake}. While BanglaFake provides a valuable benchmark for Bengali speech forensics, the synthetic samples originate predominantly from a single generation family, potentially encouraging detectors to learn generator-specific artifacts rather than generalized spoofing characteristics.

To establish a historical baseline, we additionally consider the dataset used by Ayan \emph{et al.} in their LSTM-WaveNet study of Bangla deepfake detection \cite{bangla_lstm_wavenet}. This dataset enables direct comparison with prior work and facilitates replication experiments.

\subsection{Construction of the Augmented Evaluation Dataset}

To assess cross-generator robustness, we curate an augmented evaluation dataset by combining real speech samples with synthetic utterances generated from previously unseen commercial text-to-speech systems.

Specifically, we generate synthetic Bengali speech using:

\begin{itemize}
\item \textbf{Google Gemini TTS}
\item \textbf{Crikk AI TTS}
\end{itemize}

For each generated sample, Bengali text prompts were selected to maximize linguistic diversity while avoiding duplication with training utterances. Audio files were manually verified to remove corrupted outputs, truncated speech, and generation failures.

The resulting evaluation set contains synthetic speech originating from multiple synthesis families, including VITS-derived speech from BanglaFake and proprietary neural TTS systems from Gemini and Crikk. Unlike conventional source-constrained evaluation, this mixed-source benchmark exposes models to substantial generator diversity and therefore provides a more realistic approximation of deployment conditions.


\subsection{Dataset Partitioning}

The augmented benchmark is partitioned into training, validation, and test subsets using stratified sampling to preserve source and class distributions. Following an approximate 70:15:15 split, the final partitions contain 5,117 training samples, 1,097 validation samples, and 1,097 test samples. Figure~\ref{fig:dataset_split} illustrates the resulting distribution across subsets.

\begin{figure}[t]
\centering

\begin{tikzpicture}
\begin{axis}[
    ybar stacked,
    width=\columnwidth,
    height=6cm,
    ymin=0,
    ymax=5500,
    ylabel={Number of Samples},
    symbolic x coords={Train,Validation,Test},
    xtick=data,
    legend style={
        at={(0.5,-0.25)},
        anchor=north,
        legend columns=3,
        font=\footnotesize
    },
    ymajorgrids=true,
    grid style=dashed,
    enlarge x limits=0.2
]

% BanglaFake Real
\addplot coordinates {
(Train,1944)
(Validation,416)
(Test,416)
};

% LSTM Real
\addplot coordinates {
(Train,810)
(Validation,95)
(Test,95)
};

% BanglaFake Deepfake
\addplot coordinates {
(Train,810)
(Validation,95)
(Test,95)
};

% Crikk Deepfake
\addplot coordinates {
(Train,810)
(Validation,95)
(Test,95)
};

% Gemini Deepfake
\addplot coordinates {
(Train,738)
(Validation,86)
(Test,87)
};

% LSTM Deepfake
\addplot coordinates {
(Train,810)
(Validation,95)
(Test,95)
};

\legend{
BanglaFake Real,
LSTM Real,
BanglaFake Deepfake,
Crikk Deepfake,
Gemini Deepfake,
LSTM Deepfake
}

\end{axis}
\end{tikzpicture}

\caption{
Class distribution across the train, validation, and test partitions.
A stratified split was employed to preserve the original class
proportions in all subsets.
}
\label{fig:dataset_split}
\end{figure}


\subsection{Preprocessing}

All audio recordings are standardized through resampling, mono conversion, and duration normalization to ensure consistency across sources. Speaker-aware and source-aware stratified splits are employed to reduce information leakage. For the proposed WavLM+AASIST model, RawBoost-inspired augmentations
\cite{rawboost, tak2022sslantispoofing}, including additive noise,
reverberation, and compression, are applied during training to improve
robustness under channel and generator variability.

\subsection{Proposed WavLM--AASIST Architecture}

The proposed detector combines a self-supervised speech representation model with a graph-attention-based anti-spoofing backend. The overall architecture is illustrated in Figure~\ref{fig:architecture}.

\begin{figure}[t]
\centering
% \resizebox{\columnwidth}{!}{%
% \begin{tikzpicture}[
%   block/.style={
%     draw,
%     rounded corners=6pt,
%     align=center,
%     minimum height=0.9cm,
%     font=\scriptsize,
%     fill=gray!10
%   },
%   flow/.style={-Latex, thick}
% ]

% % Main pipeline
% \node[block, minimum width=3.4cm] (aug) at (0,2.0)
% {RawBoost Augmentation\\(Noise, Reverb, Compression)};

% \node[block, minimum width=2.0cm, fill=blue!10] (raw) at (0,0.5)
% {Raw Audio\\Waveform};

% \node[block, minimum width=3.5cm, fill=green!15] (wavlm) at (4.0,0.5)
% {WavLM-Large (Frontend)\\Fine-tuned Top 12 Layers};

% \node[block, minimum width=2.3cm, fill=orange!12] (aasist) at (4.0,-1.1)
% {Modified\\AASIST};

% \node[block, minimum width=2.8cm, fill=red!8] (pred) at (7.3,-1.1)
% {Deepfake / Real\\Prediction};

% % AASIST internals
% \node[block, minimum width=4.8cm, fill=gray!20] (bottom) at (4.0,-2.7)
% {Temporal Conv + Graph Attention + OC-Softmax};

% % Connections
% \draw[flow,dashed] (aug) -- (raw);
% \draw[flow] (raw) -- (wavlm);
% \draw[flow] (wavlm) -- (aasist);
% \draw[flow] (aasist) -- (pred);

% % Missing arrow
% % \draw[flow] (bottom) -- (aasist);

% \end{tikzpicture}
% }
\includegraphics[width=0.5\textwidth]{figures/wavlm-architecture.png}
\caption{
Overview of the proposed WavLM-Large + modified AASIST pipeline. RawBoost augmentation is applied during training, followed by feature extraction using WavLM-Large and classification using a modified AASIST backend.
}
\label{fig:architecture}
\end{figure}

The frontend employs WavLM-Large, a transformer-based self-supervised speech model pretrained on large-scale speech corpora. Unlike handcrafted features such as MFCCs, WavLM preserves rich temporal,
spectral, and contextual information directly from the waveform
representation \cite{wavlm}. Recent studies have demonstrated that
self-supervised speech embeddings provide superior transferability and
robustness in spoofing detection tasks
\cite{tak2022sslantispoofing, wang2022sslfrontends, urdu2026}.

The extracted embeddings are subsequently processed by a modified AASIST backend \cite{aasist}. AASIST models spectro-temporal dependencies using graph attention mechanisms, enabling the detector to capture subtle synthesis artifacts distributed across time and frequency dimensions.

Finally, an OC-Softmax-inspired classification layer is employed to increase the separation between bona fide and spoofed samples, producing the final prediction score.

\subsection{Training Protocol}

The WavLM frontend is initialized using publicly available pretrained weights, while the upper transformer layers are fine-tuned during training. Differential learning rates are employed, with smaller learning rates assigned to pretrained layers and larger learning rates assigned to the classification backend.

Optimization is performed using AdamW with early stopping based on validation performance. Model selection is guided using multiple metrics, including Accuracy,
ROC-AUC, F1-score, and Equal Error Rate (EER), since threshold-independent
ranking quality is particularly important in forensic applications
\cite{wang2020asvspoof2019, liu2023asvspoof2021}.

\subsection{Statistical Analysis}

To determine whether observed performance differences arise from genuine architectural improvements rather than random variation, statistical significance testing is performed on extracted acoustic features and model outputs.

Feature distributions are first examined using normality tests. Depending on distributional assumptions, either parametric or non-parametric hypothesis tests are employed to compare real and synthetic speech characteristics. Effect sizes are reported alongside significance values to quantify practical relevance.

For model-level comparisons, performance metrics obtained across multiple experimental runs are evaluated using paired statistical tests. Unless otherwise stated, statistical significance is assessed at $\alpha = 0.05$.

The detailed results of these analyses are presented in Section~\ref{sec:statistical_validation}.


\section{Evaluation}
\label{sec:evaluation}

\subsection{Experimental Validation on Existing Benchmarks}

We first reproduced the MFCC-LSTM detector proposed in prior work \cite{bangla_lstm_wavenet}. On the original dataset, our implementation achieved approximately $99\%$ accuracy, substantially exceeding the $89\%$ reported in the source paper. Evaluating the same model on BanglaFake \cite{banglafake} yielded a similarly high accuracy of approximately $98\%$.

These results suggest that the detector can effectively distinguish real and synthetic speech under standard train-test splits. However, both datasets contain synthetic samples generated from a single synthesis pipeline, raising the possibility that the model learns generator-specific artifacts rather than generalizable characteristics of synthetic speech.

\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=18pt,
    ymin=80,
    ymax=100,
    ylabel={Accuracy (\%)},
    symbolic x coords={Paper,Reproduced-Orig,Reproduced-BanglaFake},
    xtick=data,
    xticklabel style={
        rotate=20,
        anchor=east,
        font=\footnotesize
    },
    nodes near coords,
    every node near coord/.append style={
        font=\footnotesize
    },
    enlarge x limits=0.25,
    width=\columnwidth,
    height=5.4cm,
    grid=major,
    ylabel style={font=\small},
    tick label style={font=\footnotesize},
    label style={font=\small},
    axis line style={black!60},
    grid style={gray!25},
]
\addplot[
    fill=blue!65!black,
    draw=blue!90!black
]
coordinates {
    (Paper,89.2)
    (Reproduced-Orig,99.67)
    (Reproduced-BanglaFake,98.69)
};
\end{axis}
\end{tikzpicture}
\caption{
Replication study of the MFCC-LSTM pipeline. Reproducing the original architecture yields substantially higher accuracy than reported in the source publication on both the original dataset and BanglaFake.
}
\label{fig:lstm_reproduction}
\end{figure}

However, both evaluations employ training and testing samples originating from the same generator distribution. Consequently, the detector may exploit generator-specific artifacts rather than learning a general boundary between authentic and synthetic speech.


\subsection{Evaluation on the Augmented Multi-Generator Dataset}

To evaluate robustness under generator shift, we constructed an augmented corpus containing deepfakes generated by four sources: BanglaFake (VITS), the Coqui-based LSTM dataset generator, CRIKK TTS, and Gemini TTS. This setting better reflects realistic deployment scenarios in which
detectors encounter previously unseen synthesis systems
\cite{muller2022generalize, liu2023asvspoof2021}.

Table~\ref{tab:main-results} summarizes the results. Compared with the near-perfect performance observed on homogeneous datasets, both detectors experience substantial degradation. The MFCC-LSTM model drops to $58.13\%$ accuracy and a ROC-AUC of $0.664$, while the proposed WavLM+AASIST detector achieves a slightly higher accuracy of $61.17\%$ and a considerably stronger ROC-AUC of $0.819$, indicating better discrimination under generator mismatch.

\begin{table}[t]
\centering
\caption{Performance on the augmented multi-generator evaluation dataset.}
\label{tab:main-results}
\begin{tabular}{l c c}
\hline
Model & Accuracy (\%) & ROC-AUC \\
\hline
LSTM & 58.13 & 0.664 \\
WavLM+AASIST & 61.17 & 0.819 \\
\hline
\end{tabular}
\end{table}

To further investigate the behavior of the two detectors, Figure~\ref{fig:generator_confidence} reports the average fake probability assigned to samples from different generators. A robust detector should assign consistently high fake probabilities to synthetic speech regardless of the underlying generation method.

\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
ybar,
bar width=8pt,
width=\columnwidth,
height=5.2cm,
ymin=0,
ymax=1.05,
ylabel={Mean Fake Probability},
symbolic x coords={
BanglaFake,
LSTM,
CRIKK,
Gemini
},
xtick=data,
enlarge x limits=0.15,
legend style={
at={(0.5,1.02)},
anchor=south,
legend columns=2,
draw=none,
font=\footnotesize
},
ylabel style={font=\small},
tick label style={font=\footnotesize},
grid=major,
grid style={gray!20}
]

\addplot[
fill=blue!70!black,
draw=blue!90!black
]
coordinates {
(BanglaFake,0.995)
(LSTM,0.061)
(CRIKK,0.009)
(Gemini,0.055)
};

\addplot[
fill=orange!80!black,
draw=orange!90!black
]
coordinates {
(BanglaFake,0.573)
(LSTM,0.377)
(CRIKK,0.269)
(Gemini,0.271)
};

\legend{MFCC-LSTM,WavLM+AASIST}

\end{axis}
\end{tikzpicture}
\caption{
Mean fake probability assigned to deepfake samples from different synthesis systems. The MFCC-LSTM detector assigns extremely high confidence to BanglaFake samples but produces near-zero fake probabilities for unseen generators (LSTM-TTS, CRIKK, and Gemini), indicating strong source dependence. In contrast, WavLM+AASIST yields substantially more consistent confidence estimates across generators, suggesting improved robustness to source shifts.
}
\label{fig:generator_confidence}
\end{figure}

The MFCC-LSTM detector exhibits a strong dependence on the generator observed during training, assigning an average fake probability of $0.995$ to BanglaFake samples but only $0.009$ and $0.061$ to CRIKK and Gemini deepfakes, respectively. In contrast, WavLM+AASIST produces substantially more consistent confidence estimates across generators, suggesting that it relies less on generator-specific artifacts and learns more transferable representations.

Overall, these results reveal a substantial generalization gap for the MFCC-LSTM baseline. While it achieves near-perfect performance on source-constrained benchmarks, its effectiveness collapses when generator diversity is introduced. The proposed WavLM+AASIST detector demonstrates improved robustness under generator shift, motivating further analysis through source-aware ablation studies.


\section{Ablation Study}
\label{sec:lofso}

To further investigate detector robustness under distribution shift, we conducted a Leave-One-Fake-Source-Out (LOFSO) evaluation. In each experiment, one synthesis source was excluded from training and used exclusively during testing. This protocol directly measures the ability of a detector to generalize to previously unseen generators rather than relying on artifacts specific to generators observed during training.

Table~\ref{tab:lofso_results} summarizes the results for both the MFCC-LSTM baseline and the proposed WavLM+AASIST detector.

\begin{table}[t]
\centering
\caption{LOFSO evaluation results.}
\label{tab:lofso_results}
\begin{tabular}{lcccc}
\toprule
\multirow{2}{*}{Held-Out Source} &
\multicolumn{2}{c}{LSTM} &
\multicolumn{2}{c}{WavLM+AASIST} \\
\cmidrule(lr){2-3}
\cmidrule(lr){4-5}
& Acc. & AUC & Acc. & AUC \\
\midrule
Full Dataset & 0.930 & 0.977 & 0.852 & 0.991 \\
LSTM-TTS     & 0.659 & 0.903 & 0.532 & 0.951 \\
Gemini       & 0.624 & 0.684 & 0.627 & 0.909 \\
CRIKK        & 0.500 & 0.718 & 0.501 & 0.972 \\
BanglaFake   & 0.458 & 0.336 & 0.464 & 0.367 \\
\midrule
Average Holdout & 0.560 & 0.660 & 0.531 & 0.800 \\
\bottomrule
\end{tabular}
\end{table}

Several important observations emerge from the LOFSO experiments.

First, the MFCC-LSTM detector exhibits a substantial reduction in performance when evaluated on unseen synthesis systems. While the model achieves an AUC of 0.977 under conventional source-matched evaluation, the average holdout AUC decreases to 0.660, indicating strong dependence on generator-specific acoustic artifacts.

Second, unseen generators vary considerably in difficulty. LSTM-TTS remains relatively easy to detect, yielding an AUC of 0.903. In contrast, Gemini and CRIKK produce lower AUC values. The most challenging scenario occurs when BanglaFake is excluded during training, where the AUC drops to 0.336. This observation suggests that BanglaFake contains synthesis characteristics that are not adequately represented by the remaining generators.

Finally, the proposed WavLM+AASIST detector demonstrates substantially stronger robustness under generator shift when evaluated using ROC-AUC. Although its average holdout accuracy (0.531) is slightly lower than that of the MFCC-LSTM baseline (0.560), the model achieves a markedly higher average holdout AUC of 0.800 compared with 0.660 for MFCC-LSTM, representing a relative improvement of approximately 21%. The advantage is particularly pronounced for the CRIKK, Gemini, and LSTM-TTS holdout scenarios, where WavLM+AASIST consistently maintains stronger discrimination between real and synthetic speech.

Taken together, these results reinforce the findings of Section~\ref{sec:evaluation}. High performance on conventional train-test splits does not necessarily imply robust deepfake detection. Instead, much of the apparent success of MFCC-LSTM models can be attributed to learning generator-specific cues rather than generalized indicators of speech synthesis. In contrast, the superior holdout AUC achieved by WavLM+AASIST suggests that self-supervised speech representations capture more transferable characteristics of synthetic speech, enabling improved discrimination under cross-generator distribution shifts.


\section{Understanding Cross-Generator Generalization}
\label{sec:statistical_validation}

The results presented in Sections~\ref{sec:evaluation} and~\ref{sec:lofso} reveal a substantial discrepancy between source-matched and source-shifted performance. To better understand the origin of this degradation, we analyze both handcrafted acoustic features and learned representation spaces.

Specifically, we investigate three questions:

\begin{enumerate}
    \item Do measurable acoustic differences exist between real and synthetic speech?
    \item Are these differences primarily associated with authenticity or generator identity?
    \item Can representation-level analyses explain the superior robustness of WavLM+AASIST?
\end{enumerate}

\subsection{Acoustic Feature Analysis}
% review comment
% is it performed on the final augmented dataset?
We first performed feature-wise significance testing on the augmented multi-generator dataset by comparing real and synthetic speech samples using Mann--Whitney U tests with Benjamini--Hochberg false discovery rate correction.

Table~\ref{tab:featuretests} reports representative features exhibiting the strongest real--fake separation.

\begin{table}[t]
\centering
\caption{Representative acoustic features exhibiting significant real--fake differences.}
\label{tab:featuretests}
\begin{tabular}{lc}
\toprule
Feature & Cohen's $d$ \\
\midrule
MFCC$_{01}$ Std & -1.13 \\
MFCC$_{06}$ Std & -1.00 \\
Peak Amplitude & -1.00 \\
MFCC$_{09}$ Std & -0.97 \\
MFCC$_{12}$ Mean & -0.97 \\
\bottomrule
\end{tabular}
\end{table}

All reported features remained significant after FDR correction ($q < 0.05$), indicating that synthetic speech exhibits measurable acoustic deviations from authentic speech. The large effect sizes (Cohen's $d > 0.97$) suggest that these differences
are practically meaningful rather than merely statistically significant
\cite{cohen1988}.

However, feature-level significance alone does not guarantee the existence of a robust decision boundary that generalizes across synthesis systems. We therefore examine the structure of the feature space itself.

\subsection{Representation Space Analysis}

To determine whether representation spaces are organized primarily by
speech authenticity or by generator identity, we performed multivariate
statistical analyses on both MFCC features and WavLM embeddings. For
MFCC features, we used permutation MANOVA and report the resulting
Pillai trace statistics. For WavLM embeddings, we performed PERMANOVA
and report the proportion of explained variance ($R^2$). In all cases,
the observed effects were statistically significant ($p < 0.001$).
Table~\ref{tab:multivariate} compares the relative influence of
authenticity labels and generator sources.


% review comment
\begin{table}[t]
\centering
\caption{Comparison of authenticity-label and generator-source effects on MFCC features and WavLM embeddings. For MFCC features, the reported statistic is the Pillai trace obtained from permutation MANOVA. For WavLM embeddings, the reported effect size is PERMANOVA $R^2$ (proportion of explained variance).}
\label{tab:multivariate}
\begin{tabular}{lccc}
\toprule
Representation & Metric & Label Effect & Source Effect \\
\midrule
MFCC  & Pillai Trace & 0.550 & 2.854 \\
WavLM & PERMANOVA $R^2$ & 0.034 & 0.370 \\
\bottomrule
\end{tabular}
\end{table}

As shown in Table~\ref{tab:multivariate}, generator source exerts a
stronger influence than authenticity label in both representation
spaces. For MFCC features, the source effect (Pillai trace = 2.854) is
substantially larger than the label effect (0.550), indicating that the
feature space is organized primarily according to synthesis source.
Similarly, for WavLM embeddings, source identity explains a larger
proportion of variance ($R^2 = 0.370$) than authenticity label
($R^2 = 0.034$). These results suggest that generator-dependent
structure remains present in both representations. However, the
multivariate statistics alone do not reveal how these effects manifest
geometrically in the feature space. We therefore complement the analysis
with low-dimensional visualizations.

Figure~\ref{fig:mfcc-tsne} provides additional evidence.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/tsne_mfcc.png}
\caption{MFCC feature visualization. Samples cluster primarily according to generator source rather than a unified real--fake separation.}
\label{fig:mfcc-tsne}
\end{figure}

Rather than forming a coherent authenticity-oriented separation, MFCC representations organize primarily according to synthesis source. This observation provides a potential explanation for the poor cross-generator performance observed in the LOFSO experiments.

We next examine the representation space learned by WavLM.

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_embedding_pca.png}
\caption{PCA projections of WavLM embeddings. Left: samples colored by
authenticity label. Right: samples colored by generator source.
Generator-specific structure remains visible, but source groups exhibit
substantial overlap compared with the MFCC feature space shown in
Figure~\ref{fig:mfcc-tsne}.}
\label{fig:wavlm-pca}
\end{figure*}

% review comment
Figure~\ref{fig:wavlm-pca} provides additional insight into the
structure of the WavLM embedding space. The left panel, colored by
authenticity label, shows partial separation between real and synthetic
speech samples, indicating that the embeddings capture information
relevant to the deepfake detection task. The right panel, colored by
generator source, reveals that generator-specific structure remains
present; however, the source groups exhibit substantially greater
overlap than the distinct clusters observed in the MFCC t-SNE
visualization (Figure~\ref{fig:mfcc-tsne}).

Taken together, the statistical results in
Table~\ref{tab:multivariate} and the visualizations in
Figures~\ref{fig:mfcc-tsne} and~\ref{fig:wavlm-pca} suggest that both
representations are influenced by generator identity, but the WavLM
embedding space is less strongly partitioned by source than the MFCC
feature space. This reduced source-specific clustering provides a
plausible explanation for the improved cross-generator robustness of the
proposed WavLM+AASIST detector observed in the LOFSO experiments.

\subsection{Generator Drift and Representation Generalization}

A natural hypothesis is that generators exhibiting larger acoustic differences should also produce larger performance degradation. To test this hypothesis, we quantified generator drift using Wasserstein distances computed over MFCC feature distributions.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_drift_corr.png}
\caption{Generator drift versus LOFSO AUC degradation. Acoustic drift measured in MFCC space does not reliably predict cross-generator performance.}
\label{fig:drift_auc}
\end{figure}

Figure~\ref{fig:drift_auc} demonstrates that this relationship is weak. For example, CRIKK exhibits the largest acoustic drift but does not produce the largest degradation in detection performance. Conversely, BanglaFake yields the most severe performance reduction despite exhibiting more moderate acoustic divergence.

These results suggest that conventional MFCC statistics fail to capture the synthesis characteristics most relevant for cross-generator detection. Consequently, the poor generalization of MFCC-LSTM models cannot be explained solely by low-level acoustic differences. Instead, the evidence presented in Figures~\ref{fig:mfcc-tsne}
and~\ref{fig:wavlm-pca} indicates that representation learning plays a
central role in determining robustness under generator shift
\cite{wavlm, tak2022sslantispoofing, muller2024harder}.


\section{Discussion}

The primary finding of this study is that conventional evaluation protocols substantially overestimate the robustness of Bangla audio deepfake detectors. Under source-matched evaluation, the MFCC-LSTM baseline achieves nearly perfect performance, reproducing and even exceeding results reported in prior work. However, introducing generator diversity produces a dramatic decline in performance, revealing a substantial gap between benchmark accuracy and real-world robustness.

The LOFSO experiments further demonstrate that this degradation is not uniform across generators. Some synthesis systems remain relatively easy to detect, whereas others cause severe performance collapse. This observation suggests that current detectors are sensitive to generator-specific artifacts rather than learning a unified representation of synthetic speech.

The statistical analyses provide additional evidence supporting this interpretation. Although numerous MFCC-derived features exhibit significant real--fake differences, representation-level analyses reveal that MFCC spaces are dominated by generator-dependent structure. The t-SNE visualization demonstrates clear source-based clustering, while the drift analysis shows that simple acoustic divergence is insufficient to explain cross-generator performance.

In contrast, WavLM embeddings exhibit improved organization of authentic and synthetic speech samples across generators. Combined with the stronger ROC-AUC observed on the multi-generator benchmark, these findings suggest that self-supervised representations capture more transferable synthesis cues than handcrafted acoustic features.

Collectively, these results indicate that future Bangla audio deepfake research should prioritize source-diverse evaluation and distributional robustness rather than solely optimizing performance on homogeneous datasets. The benchmark introduced in this work represents a step toward more realistic forensic evaluation settings and highlights the importance of generator-aware validation protocols for emerging low-resource languages.


\section{Conclusion}

This work revisited Bangla audio deepfake detection from a generalization perspective. We first reproduced a previously proposed LSTM-based detector and achieved near-perfect performance on existing benchmark datasets. However, experiments on a newly constructed multi-generator corpus revealed that these impressive results do not necessarily translate to robust real-world detection.

We introduced a WavLM+AASIST based detector and performed extensive statistical analyses of both MFCC and embedding representations. The results demonstrate that MFCC-based models are highly sensitive to generator-specific artifacts, whereas WavLM embeddings provide a more consistent real-fake representation across synthesis systems.

Finally, LOFSO experiments identified substantial performance variation across unseen generators, emphasizing the importance of generator diversity in benchmark design.

These findings suggest that future research should prioritize cross-generator robustness and distributional generalization rather than solely maximizing accuracy on homogeneous datasets.

% \newpage

% \begin{thebibliography}{00}
% \bibitem{banglafake} K. Asif et al., ``BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset,'' arXiv, 2025. 
% \bibitem{bangla_lstm_wavenet} A. Ayan et al., ``Detecting Bangla DeepFake Audio: A Dual Approach Using LSTM and WaveNet,'' 2026. \bibitem{urdu2026} M. Owais et al., ``Deepfake Audio Detection in Low-Resource Languages: A Case Study of Urdu,'' IEEE Access, 2026. \bibitem{survey2024} Author(s), ``A Survey on Audio Deepfake Detection,'' Year. 
% \bibitem{review2024} Author(s), ``A Review of Modern Audio Deepfake Detection Methods,'' Year. 
% \bibitem{ahead2025} Author(s), ``Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead,'' Year. 
% \bibitem{generalize} Author(s), ``Does Audio Deepfake Detection Generalize?'' Year. 
% \bibitem{mfccml} Author(s), ``Deepfake Audio Detection via MFCC Features Using Machine Learning,'' Year. 
% \bibitem{deepfake_dl} Author(s), ``Leveraging Deep Learning Methods for Detecting Deepfake Speeches,'' Year. 
% \bibitem{shaaban2025} O. A. Shaaban and R. Yildirim, ``Audio Deepfake Detection Using Deep Learning,'' Engineering Reports, 2025. \bibitem{verbafake} Author(s), ``VerbaFake, EchoFake, and PixelFake,'' Year. 
% \bibitem{vicomtech} Author(s), ``Vicomtech Audio Deepfake Detection Study,'' Year. 

% \bibitem{rawboost}
% H. Tak, M. Kamble, J. Patino, M. Todisco, and N. Evans,
% ``RawBoost: A Raw Data Boosting and Augmentation Method Applied to Automatic Speaker Verification Anti-Spoofing,''
% in \emph{Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 2022, pp. 6382--6386.

% \bibitem{aasist}
% J.-W. Jung, H.-S. Heo, H. Tak, H.-J. Shim, J.-S. Chung,
% B.-J. Lee, and N. S. Kim,
% ``AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks,''
% in \emph{Proc. Interspeech}, 2021, pp. 636--640.

% \bibitem{wavlm}
% S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, J. Li, N. Q. K. Duong,
% X. Liu, Y. Wei, and F. Ma,
% ``WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing,''
% \emph{IEEE Journal of Selected Topics in Signal Processing},
% vol. 16, no. 6, pp. 1505--1518, 2022.

% \end{thebibliography}
\newpage
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}

