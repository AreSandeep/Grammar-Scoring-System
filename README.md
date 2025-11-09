# Grammer-Scoring-


# 🎙️ Grammar Scoring Engine from Voice Samples

This project builds an **AI-powered Grammar Scoring Engine** that evaluates spoken grammar proficiency from **voice samples (.wav)** and **transcripts (.txt)**.

The pipeline extracts linguistic, readability, and prosodic features from each sample, generates heuristic grammar scores (labels), and trains a machine learning model to predict grammar quality automatically.

---

## 🚀 Features

✅ End-to-end ML pipeline:
- Audio + Text feature extraction  
- Grammar error detection using `language_tool_python`  
- Readability metrics using `textstat`  
- Prosodic features (pitch, energy, duration, tempo) via `librosa`  
- Text fluency estimation via GPT-2 perplexity  
- Synthetic label generation (grammar quality score 0–10)  
- Regression model training with `XGBoost`  
- Inference pipeline for new voice samples  

---

## 🧠 Project Workflow

### 1️⃣ Data Preparation
You provide `.wav` files (voice samples) and optional `.txt` transcripts stored in Google Drive:

/content/drive/MyDrive/grammar_voice_samples/
├── sample1.wav
├── sample1.txt
├── sample2.wav
├── sample2.txt




If no `.txt` exists, an ASR placeholder function (`transcribe_file_whisper`) can be used for speech-to-text transcription.

---

### 2️⃣ Feature Extraction
Each audio–text pair is processed to extract:

| Category | Features |
|-----------|-----------|
| **Grammar** | `total_errors` (grammar errors via LanguageTool) |
| **Text** | `token_count`, `ppl` (GPT-2 perplexity), `flesch_kincaid`, `flesch_reading` |
| **Audio (Prosody)** | `duration`, `energy`, `f0_mean`, `f0_std`, `zcr`, `tempo` |

These features are saved to:
/content/drive/MyDrive/grammar_voice_samples_output/features.csv


---

### 3️⃣ Label Generation (Grammar Quality Score)
Since no human labels were available, a **synthetic score** was generated using a weighted formula:

\[
\text{label} = 10 \times \frac{(0.6e^{-0.2 \times \text{total_errors}} + 0.3 \times \text{readability} + 0.1e^{-0.1 \times f0_{std}}) - \min}{\max - \min}
\]

This creates a continuous grammar proficiency score between **0 (poor)** and **10 (excellent)**.

Resulting dataset is saved as:
features_with_labels.csv


---

### 4️⃣ Model Training
A regression model was trained using **XGBoost** to learn relationships between extracted features and the grammar score.

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

predict_grammar_score("path/to/sample.wav", "path/to/sample.txt")

📊 Example Output
filename	total_errors	flesch_reading	f0_std	label
000010-0014.wav	3	67.2	22.5	6.8
000010-0015.wav	8	45.3	30.2	3.4


🧩 Technologies Used
Category	Tools
Language Processing	language_tool_python, textstat, transformers (GPT-2)
Audio Processing	librosa
Machine Learning	scikit-learn, xgboost
Environment	Google Colab, Python 3.10
Data Storage	Google Drive integration
