import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import torch

# Utility: Ensure directory existence
def ensure_dir(directory: str) -> None:
    '''
    Ensure that a directory exists. If it does not, create it.
    '''
    os.makedirs(directory, exist_ok=True)

# Load audio using pydub (in target sample rate)
def load_audio(file_path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    '''
    Load audio from a file, resample to a target sample rate, and normalize the audio signal.
    
    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sampling rate for the audio.

    Returns:
        tuple: Preprocessed audio signal as a tensor and its sampling rate.
    '''
    # Load audio using pydub
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr) #.set_channels(1)  ensure mono
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)  # Normalize to [-1, 1]
        return torch.tensor(samples).unsqueeze(0), target_sr
    except Exception as e:
        print(f'Error loading {file_path}: {e}')
        return torch.tensor([]), 0

def preprocess_audio(file_path: str, target_sr: int = 16000, mono: bool = True, silence_threshold: float = -30.0, high_pass_freq: int = 50) -> str:
    '''
    Preprocess an audio file by applying standard cleaning techniques such as 
    resampling, normalization, noise reduction, and trimming silence. The cleaned 
    file is saved with a '_processed' suffix.

    Args:
        file_path (str): Path to the original audio file.
        target_sr (int): Target sampling rate for the audio (default: 16000 Hz).
        mono (bool): Whether to convert the audio to mono (default: True).
        silence_threshold (float): dBFS threshold for silence removal (default: -30.0 dBFS).
        high_pass_freq (int): Frequency threshold for high-pass filtering (default: 50 Hz).

    Returns:
        str: Path to the processed audio file. Returns None if an error occurs.
    '''
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Convert to mono if specified
        if mono and audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample if the current frame rate does not match the target sample rate
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        # Normalize volume based on the file's dBFS level
        audio = audio.apply_gain(-audio.dBFS)

        # Trim leading and trailing silence based on the silence threshold
        audio = audio.strip_silence(silence_threshold=silence_threshold)

        # Apply high-pass filters to remove unwanted frequency noise
        audio = audio.high_pass_filter(high_pass_freq)

        # Define the file path for the processed audio
        new_file_path = os.path.splitext(file_path)[0] + '_processed.mp3'

        # Save the processed audio file
        audio.export(new_file_path, format='mp3')

        return new_file_path
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return file_path


# Extract audio features
def extract_audio_features(file_path: str, target_sr: int = 16000) -> dict:
    '''
    Extract relevant audio features such as duration, MFCCs, and spectral centroid from an audio file.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sampling rate.

    Returns:
        dict: Extracted features with keys such as 'duration', 'mfccs', and 'spectral_centroid'.
    '''
    # Load audio
    audio, sr = load_audio(file_path, target_sr)
    
    if audio is not None:
        # Remove unnecessary dimensions and convert to numpy array
        y = audio.squeeze().numpy()
        
        # Compute and return various audio features
        return {
            'duration': librosa.get_duration(y=y, sr=sr),                               # Total duration of the audio in seconds
            'mfccs': librosa.feature.mfcc(y=y, sr=sr).mean(axis=1),                     # Mean MFCCs values across time
            'energy': np.sum(y**2),                                                     # Sum of squared values (energy of the signal)
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).mean(),         # Mean zero-crossing rate
            # 'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),  # Mean spectral centroid
            # 'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),# Mean spectral bandwidth
            # 'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr).mean(),  # Mean spectral contrast
            # 'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),    # Mean spectral rolloff
            # 'chroma_stft': librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)         # Mean chroma feature values
        }
    
    # Return audio values or NaN values if audio cannot be loade
    return {key: np.nan for key in ['duration', 'mfccs', 'energy', 'zero_crossing_rate', 
                                    # 'spectral_centroid', 'spectral_bandwidth', 
                                    # 'spectral_contrast', 'spectral_rolloff', 'chroma_stft'
                                    ]}


# Preprocess entire dataset
def preprocess_data(df: pd.DataFrame, audio_folder: str) -> pd.DataFrame:
    '''
    Preprocess all audio files in the dataset by resampling, normalizing, and extracting features.
    
    Args:
        df (pd.DataFrame): DataFrame containing filenames of audio files.
        audio_folder (str): Directory containing audio files.

    Returns:
        pd.DataFrame: Updated DataFrame with paths to processed files and extracted features.
    '''
    df['processed_path'] = df['filename'].apply(lambda x: preprocess_audio(os.path.join(audio_folder, x)))

    # Drop rows with missing values
    df = df.dropna(subset=['processed_path'])

    # Extract audio features
    audio_features = df['processed_path'].apply(extract_audio_features)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(audio_features.tolist())], axis=1)

# Visualize audio features
def visualize_audio_features(df: pd.DataFrame, audio_folder: str, sample_num=3) -> None:
    '''
    Visualize waveform, spectrogram, and MFCCs for randomly sampled audio files from the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing audio filenames.
        audio_folder (str): Directory containing audio files.
        sample_num (int): Number of audio files to sample and visualize. Default is 3.
    '''
    # Randomly sample audio files from the DataFrame
    sample_files = df['filename'].sample(sample_num, random_state=123).tolist()
    
    for file in sample_files:
        # Construct the full file path
        file_path = os.path.join(audio_folder, file)
        filename = file_path.split('/')[-1]  # Extract only the filename for display purposes
        
        # Load the audio file
        audio, sr = load_audio(file_path)
        
        # Skip visualization if audio loading fails
        if audio is None:
            continue

        # Waveform Visualization
        y = audio.squeeze().numpy()  # Remove unnecessary dimensions and convert to numpy array
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f'Waveform of {filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

        # Spectrogram Visualization
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # Compute spectrogram in dB
        plt.figure(figsize=(7, 2))
        librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', cmap='cool')
        plt.colorbar(format='%+2.0f dB')  # Add color bar with decibel format
        plt.title(f'Spectrogram of {filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()