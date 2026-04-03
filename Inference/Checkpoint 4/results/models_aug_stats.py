"""
This file loads the model_comparison_predictions.csv file and calculates the following evaluation metrics for each model:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
The results are then printed in a tabular format for easy comparison.

input file structure:
filepath,folder,true_label,lstm_pred,lstm_prob_fake,lstm_prob_real,wave2vec_pred,wave2vec_prob_fake,wave2vec_prob_real,wave2vec_score,xlsr_pred,xlsr_prob_fake,xlsr_prob_real,xlsr_score,agree_lstm_wave2vec,agree_lstm_xlsr,agree_wave2vec_xlsr,models_agree_all
/kaggle/input/datasets/prithuanan/bangla-audio-deepfake-detection/Data/baglafake_deepfake/deepfake_data_mozilla__deepfake_wav__common_voice_s1_123.wav,baglafake_deepfake,1,1,0.9996131062507629,0.0003869648789986968,1,0.6314597725868225,0.3685402274131775,-0.5384843349456787,1,0.5031927824020386,0.4968072175979614,-0.012771256268024445,1,1,1,1

write the output in the file models_aug_stats.txt in the following format:
Model: LSTM (MFCC-based)
Accuracy: 0.5000
Precision (Real/Fake): 0.5000 / 0.0000
Recall (Real/Fake): 1.0000 / 0.0000
F1 Score (Real/Fake): 0.6667 / 0.0000
ROC-AUC Score: N/A

Model: WaveNet (Raw Audio)
Accuracy: 0.5000
Precision (Real/Fake): 0.5000 / 0.0000
Recall (Real/Fake): 1.0000 / 0.0000
F1 Score (Real/Fake): 0.6667 / 0.0000
ROC-AUC Score: N/A

Model: WavLM + AASIST
Accuracy: 0.5000
Precision (Real/Fake): 0.5000 / 0.0000
Recall (Real/Fake): 1.0000 / 0.0000
F1 Score (Real/Fake): 0.6667 / 0.0000
ROC-AUC Score: N/A
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Load the CSV file
df = pd.read_csv('results/model_comparison_predictions.csv')

# True labels
y_true = df['true_label']

# Model configurations: (name, pred_column, prob_column)
models = [
    ('LSTM (MFCC-based)', 'lstm_pred', 'lstm_prob_fake'),
    ('WaveNet (Raw Audio)', 'wave2vec_pred', 'wave2vec_prob_fake'),
    ('WavLM + AASIST', 'xlsr_pred', 'xlsr_prob_fake')
]

# Open the output file
with open('models_aug_stats.txt', 'w') as f:
    for name, pred_col, prob_col in models:
        y_pred = df[pred_col]
        prob = df[prob_col]
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # ROC-AUC
        try:
            auc = roc_auc_score(y_true, prob)
            auc_str = f'{auc:.4f}'
        except ValueError:
            auc_str = 'N/A'
        
        # Write to file
        f.write(f'Model: {name}\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Precision (Real/Fake): {prec[0]:.4f} / {prec[1]:.4f}\n')
        f.write(f'Recall (Real/Fake): {rec[0]:.4f} / {rec[1]:.4f}\n')
        f.write(f'F1 Score (Real/Fake): {f1[0]:.4f} / {f1[1]:.4f}\n')
        f.write(f'ROC-AUC Score: {auc_str}\n\n')

print("Evaluation metrics have been calculated and written to models_aug_stats.txt")