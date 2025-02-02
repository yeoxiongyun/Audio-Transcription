# Finetune an Automatic Speech Recognition (ASR) AI model

> Indented block

> Indented block

## 0. Setup

**Package Installations**

This code runs shell commands and install relevant python packages and libraries.

**GPU Configurations**

This code checks for GPU availability and prints their details. If a GPU is available, it compares the execution time of a matrix multiplication operation on both the CPU and GPU.

#### Library Installations

#### Library Imports

## 1. Data Collection

For this project, we will be using the [Common Voice](https://www.kaggle.com/datasets/mozillaorg/common-voice/data) dataset sourced from Kaggle. The audio clips for each subset are stored as mp3 files in folders with the same naming conventions as their corresponding CSV files. For instance, all audio data from the valid train set is stored in the folder `cv-valid-train`, alongside the `cv-valid-train.csv` metadata file.

### **CSV File Structure**

Each row in the CSV file represents a single audio clip and contains the following information:

- **`filename`**: Relative path of the audio file.
- **`text`**: Supposed transcription of the audio.
- **`up_votes`**: Number of people who confirmed the audio matches the text.
- **`down_votes`**: Number of people who reported the audio does not match the text.
- **`age`**: Age of the speaker, if reported:
  - `teens`: `< 19`
  - `twenties`: `19 - 29`
  - `thirties`: `30 - 39`
  - `fourties`: `40 - 49`
  - `fifties`: `50 - 59`
  - `sixties`: `60 - 69`
  - `seventies`: `70 - 79`
  - `eighties`: `80 - 89`
  - `nineties`: `> 89`
- **`gender`**: Gender of the speaker, if reported:
  - `male`
  - `female`
  - `other`
- **`accent`**: Accent of the speaker, if reported:
  - `us`: `United States English`
  - `australia`: `Australian English`
  - `england`: `England English`
  - `canada`: `Canadian English`
  - `philippines`: `Filipino`
  - `hongkong`: `Hong Kong English`
  - `indian`: `India and South Asia (India, Pakistan, Sri Lanka)`
  - `ireland`: `Irish English`
  - `malaysia`: `Malaysian English`
  - `newzealand`: `New Zealand English`
  - `scotland`: `Scottish English`
  - `singapore`: `Singaporean English`
  - `southatlandtic`: `South Atlantic (Falkland Islands, Saint Helena)`
  - `african`: `Southern African (South Africa, Zimbabwe, Namibia)`
  - `wales`: `Welsh English`
  - `bermuda`: `West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)`

### **Acknowledgments**
This dataset was compiled by Michael Henretty, Tilman Kamp, Kelly Davis, and The Common Voice Team.

## 2. Data Pre-processing

#### Data Cleaning
 * remove NaN or null values
 * remove irrelevant/ unnecessary data

#### Train-Validation Split

#### Preliminary Data Visualisation
* Waveform Visualization
* Mel-frequency cepstral coefficients (MFCCs) / Spectrogram

#### Data Extraction (Feature Engineering)

## 2. Data Extraction (Feature Engineering)

## 5. Model Choice

The chosen ASR AI Model is [wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h). This model is developed by Facebook and pretrained and fine-tuned on Librispeech dataset on 16kHz sampled speech audio.

## 6. Model Training

## 7. Model Evaluation

For evaluating models like `Wav2Vec2ForCTC` in tasks such as Automatic Speech Recognition (ASR), the following metrics are typically used:

### 1. Word Error Rate (WER)
- **Definition**: Measures the percentage of incorrectly predicted words in the transcript.
- **Formula**:
  $WER = \frac{S + D + I}{N}$
  where:
  - \( S \): Number of substitutions (wrong word predicted).
  - \( D \): Number of deletions (missing words in prediction).
  - \( I \): Number of insertions (extra words in prediction).
  - \( N \): Total number of words in the reference transcript.
- **Usage**:
  - Lower WER indicates better model performance.
  - It is the most common metric for ASR systems.

---

The pre-trained implementation from Pytorch will serve as a benchmark for us to determine how good the current implementation of the model is.

## 8. Model Understanding (Explainability)