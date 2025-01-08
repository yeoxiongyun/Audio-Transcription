import requests
import sys

# Loading Audio (Client Side Implementation)
def load_audio(audio_path: str) -> tuple:
    '''
    Load the audio file to send to the API.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        tuple: Audio file name and binary content.
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
    print('Testing audio speech recognition...')
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

            print(f'Audio Transcript: {transcript}')
            print(f'Audio Duration: {duration}')
        else:
            print(f'Error: {response.status_code} - {response.text}')
    except Exception as e:
        print(f'Failed to test ASR API: {e}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python test_client_script.py [audio_dir]')
        sys.exit(1)
    
    # print(sys.argv)
    audio_dir = sys.argv[1]
    test_ping()
    test_asr(audio_dir)