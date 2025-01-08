import requests
import sys
import csv
import os
from typing import Tuple

def load_audio(audio_path: str) -> Tuple[str, bytes, str]:
    '''
    Load the audio file to send to the API.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        tuple: Audio file name, binary content, and MIME type.
    '''
    with open(audio_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    return audio_path, audio_bytes, 'audio/mp3'

def test_ping():
    '''
    Test the ping endpoint to check if the API is running.
    '''
    print('Testing API Health...')
    url = 'http://localhost:8001/ping'
    
    # Send GET request to /ping endpoint
    response = requests.get(url)
    
    # Process the response
    if response.status_code == 200:
        print('API is running...')
        print(f'API message: {response.json().get("message", "No message provided")}')
    else:
        # print(f'Error: {response.status_code} - {response.text}')
        print(f'Failed to connect to the API. Status code: {response.status_code}')
        print(f'Error: {response.text}')

    
def test_asr(audio_dir):
    '''
    Test the ASR API with a single audio file.
    '''
    url = 'http://localhost:8001/asr'

    try:
        _, audio_bytes, content_type = load_audio(audio_dir)
        files = {'file': (audio_dir, audio_bytes, content_type)}

        # Send POST request to ASR API
        response = requests.post(url, files=files)

        if response.status_code == 200:
            response_data = response.json()
            transcript = response_data.get('transcript', '')
            duration = response_data.get('duration', '')

            return transcript, duration
        else:
            print(f'Error: {response.status_code} - {response.text}')
            return 'Error processing file', '00:00'
    except Exception as e:
        print(f'Failed to test ASR API: {e}')
        return 'Error processing file', '00:00'

def process_csv(csv_path: str, audio_base_dir: str, batch_size: int = 100):
    '''
    Process the CSV file, update with transcripts and durations, and save changes in batches.
    Args:
        csv_path (str): Path to the CSV file.
        audio_base_dir (str): Base directory containing the audio files.
        batch_size (int): Number of rows to process before saving.
    '''
    updated_rows = []
    output_csv_path = csv_path.replace('.csv', '_updated.csv')

    with open(csv_path, mode='r', encoding='utf-8') as infile, \
         open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        # Add 'generated_text' column if not present
        fieldnames = reader.fieldnames # Existing columns
        if 'generated_text' not in fieldnames: # type: ignore
            fieldnames.append('generated_text') # type: ignore
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames) # type: ignore
        writer.writeheader()

        for i, row in enumerate(reader, start=1):
            audio_file_path = os.path.join(audio_base_dir, row['filename'])
            if os.path.exists(audio_file_path):
                transcript, duration = test_asr(audio_file_path) # Get transcript and duration from ASR API
                row['duration'] = duration                       # Populate 'duration' column
                row['generated_text'] = transcript               # Add transcript to the new column
            # Handle missing files
            else:
                row['duration'] = '00:00'
                row['generated_text'] = f'File [{audio_file_path}] not found'
            
            updated_rows.append(row)

            # Save to disk every 'batch_size' rows
            if i % batch_size == 0:
                print(i)
                writer.writerows(updated_rows)
                updated_rows = []  # Clear buffer for next batch
        
        # Write any remaining rows
        if updated_rows:
            writer.writerows(updated_rows)

    print(f'Updated CSV saved to: {output_csv_path}')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 cv-decode.py [csv_path] [audio_base_dir]')
        sys.exit(1)

    csv_path = sys.argv[1]
    audio_base_dir = sys.argv[2]

    test_ping()
    process_csv(csv_path, audio_base_dir)