from flask import Flask, request, jsonify
import os
import torch
from pydub import AudioSegment
from transformers import AutoProcessor, AutoModelForCTC
from pathlib import Path

import multiprocessing
# multiprocessing.set_start_method('fork', force=True) -- is primarily for CPU workloads.
multiprocessing.set_start_method('spawn', force=True)

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG)


# Current Working Director, os.getcwd(): /workspace

app = Flask(__name__)

@app.before_request
def validate_api_key():
    expected_api_key = os.getenv('API_KEY')
    provided_api_key = request.headers.get('API-Key')   # key = request.headers.get('Authorization')
     #print(f'Expected API Key: {expected_api_key}')
    # print(f'Provided API Key: {provided_api_key}') 
    
    if provided_api_key != expected_api_key:
        return jsonify({'Error': 'Invalid API key'}), 401

# Check for MPS backend support (Metal Performance Shaders)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {device}')

# Specify the directory where the model and processor are saved
model_dir = 'model'
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCTC.from_pretrained(model_dir, local_files_only=True).to(device) # model = AutoModelForCTC.from_pretrained(model_dir)

# Loading Audio (Server Side Implementation)
def load_audio(audio_path: str) -> tuple[torch.Tensor, int, str]:
    '''
    Load and preprocess the audio file for ASR.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        torch.Tensor: Preprocessed audio tensor.
        int: Sampling rate of the audio.
        str: Path to the converted audio file.
    '''
    audio = AudioSegment.from_file(audio_path, format='mp3')
    sample_rate = audio.frame_rate

    # Convert to 16kHz (downmix to mono: .set_channels(1))
    if sample_rate != 16000 or audio.channels > 1:
        audio = audio.set_frame_rate(16000).set_channels(1)
        converted_path = os.path.join('audio', 'converted_audio_16kHz.mp3')
        audio.export(converted_path, format='mp3')
        audio_path = converted_path
        sample_rate = 16000
    
    # Convert audio samples to tensor
    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples, dtype=torch.float32) / (2 ** 15)

    return waveform.unsqueeze(0), sample_rate, audio_path

@app.route('/ping', methods=['GET'])
def ping():
    '''
    Health check API to verify the service is running.
    '''
    return jsonify({'message': 'pong'}), 200


@app.route('/asr', methods=['POST'])
def asr():
    '''
    Hosted inference API for automatic speech recognition (ASR).
    Input:
        - file: binary MP3 audio file (multipart/form-data)
    Output:
        - transcript: Transcribed text from audio
        - duration: Duration of the audio file in seconds
    '''
    if 'file' not in request.files:
        return jsonify({'Error': 'No audio file provided'}), 400
    
    # Get the audio file from the request
    audio_file = request.files['file']
    filename = audio_file.filename

    # Resolves to an absolute path
    audio_path = Path(filename).resolve() # type: ignore

    try:
        # Load and preprocess the audio
        waveform, sample_rate, _ = load_audio(audio_path) # type: ignore
        duration_seconds = waveform.size(1) / sample_rate

        # Convert duration to 'minutes:seconds' format
        minutes = int(duration_seconds) // 60
        seconds = int(duration_seconds) % 60
        duration_formatted = f'{minutes:02}:{seconds:02}'

        # Prepare inputs for the model
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors='pt', padding=True)
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Perform inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted IDs to text
        # transcript = processor.batch_decode(predicted_ids)[0]
        transcript = processor.batch_decode(predicted_ids)

        response = {
            'transcript': transcript[0],
            'duration': f'{duration_formatted}'
        }

    except Exception as e:
        logging.error(f'Failed to process audio file: {e}')
        response = {'Error': f'{str(e)}'}
    
    # Delete the processed audio file
    finally:
        # Exclude test audio files from deletion
        audio_filename = os.path.basename(audio_path)
        if audio_file.filename not in ['harvard.wav', 'dingSFX-1.mp3']:
            try:
                os.remove(audio_path)  
                logging.info(f'Deleted the original audio file: {audio_filename}')
            except OSError as delete_error:
                logging.error(f'Error deleting file {audio_filename}: {delete_error}')

    return jsonify(response), 200 if 'Error' not in response else 500

if __name__ == '__main__':
    # Run the Flask app on port 8001
    app.run(host='0.0.0.0', port=8001)