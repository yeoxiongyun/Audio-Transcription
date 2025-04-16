# Audio Transcription Pipeline


This project provides an Audio Transcription API built with Flask and TensorFlow, along with tools to test the API and preprocess data for speech recognition. It supports Docker deployment, API testing via curl or Python scripts, and speech data experimentation in Jupyter Notebooks. Use the tools provided to test, debug, and extend the ASR functionality.

```
+--------------------+       Upload MP3/Audio       +-------------------------+
|                    |----------------------------->|                         |
|   MP3 Audio File   |                              |     ASR API Endpoint    |
|  (e.g., ding.mp3)  |                              |                         |
+--------------------+                              +-------------------------+
                                                      |
                                                      v
                                           +-----------------------------+
                                           |   Speech Recognition Model  |
                                           |                             |
                                           |   Transcription Logic       |
                                           |   (TensorFlow/Python)       |
                                           |                             |
                                           +-----------------------------+
                                                      |
                                                      v
                                   +---------------------------------------+
                                   |                                       |
                                   |      JSON Response with Transcript    |
                                   |     {"transcript": "DING SOUND...",   |
                                   |        "duration": "00:02"}           |
                                   |                                       |
                                   +---------------------------------------+

+-------------------+              +-----------------------------------------+
|                   |              |                                         |
|  Client Side API  |<-------------|  Flask API (Localhost:8001 or Docker)   |
|  Tester (Curl/Py) |              |   - Health Check (/ping)                |
|                   |              |   - ASR (/asr)                          |
+-------------------+              +-----------------------------------------+
                                                      ^
                                                      |
               +--------------------------------------+------------------------------------+
               |                                                                           |
   +----------------------------+                                     +--------------------------+
   | Flask-CLI or Direct Run    |                                     | Docker Environment       |
   | python3 asr_api.py         |                                     | docker build/run         |
   +----------------------------+                                     +--------------------------+
```

## Setup Instructions

#### Step 0: Clone the GitHub Repository
Clone the project repository to your local machine:

```bash
git clone https://github.com/yeoxiongyun/Audio-Transcription
```

The folder structure should be as follows:
```
speech/
├── asr
│   ├── ...
│   ├── asr_api.py
│   ├── cv-decode.py
│   ├── test_client_script.py
│   ├── audio
│   │   ├── ...
│   ├── docker
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   ├── model
├── asr-train
├── common_voice
├── ...
```


#### Step 1: Directory Navigation

Navigate to the `docker` folder in the project repository:
```bash
cd /path/to/project/../speech/asr/docker
```

#### Step 2: Build the Docker Image

Build the Docker image from the parent folder 3 levels up:
```bash
docker build -t asr-app:1.0 -f ./Dockerfile ../../../
```

#### Step 3: Run the Docker Image

Run the API in a Docker container, exposing it on port 8001:
```bash
cd ../
docker run -p 8001:8001 asr-app:1.0
docker run --env-file docker/.env -p 8001:8001 asr-app:1.0
```

#### Optional: Run the Flask Application Directly

If you prefer running the application without Docker, execute the Flask API directly.:
```bash
cd ../
python3 asr_api.py
```

### Testing the API Endpoints

You can test the /asr endpoint using either curl commands or the provided client-side Python script. The sample audio files are sourced from the web. (e.g. [harvard.wav](https://www.kaggle.com/datasets/pavanelisetty/sample-audio-files-for-speech-recognition)) Open a new tab or window of shell. Navigate to the `asr` folder (if needed)and run the following options to send audio files to the API.


#### **Options**:
   1. **Client URL (curl)**  
      ```bash
      curl http://localhost:8001/ping
      curl -F 'file=@audio/harvard.wav' http://localhost:8001/asr
      ```
      Expected Output:
      ```bash
      {"duration":"00:18","transcript":"..."}
      ```

   2. **Python Script**  
      ```bash
      python3 test_client_script.py audio/harvard.wav
      ```
      Expected Output:
      ```bash
      Testing API Health...
      API is running...
      Audio Transcript: ...
      Audio Duration: ...
      ```

   3. **Flask Command Line Interface (Flask-CLI)**  
      ```bash
      flask --app asr_api.py run --host=0.0.0.0 --port=8001
      ```


### Transcribe Audio Files & Update CSV

To process MP3 audio files and update the CSV with transcripts, run the cv-decode.py script. This script will transcribe audio files listed in the CSV and append their transcripts and duration.

```bash
python3 cv-decode.py ../common_voice/cv-valid-dev.csv ../common_voice/cv-valid-dev/
```

Expected Output:
```
Updated CSV saved to: ../common_voice/cv-valid-dev_updated.csv
```

| Filename                         | Text                                                                                     | Up Votes | Down Votes | Age       | Gender | Accent  | Duration | Generated Text                                                                       |
|----------------------------------|-----------------------------------------------------------------------------------------|----------|------------|-----------|--------|---------|----------|-------------------------------------------------------------------------------------|
| `cv-valid-dev/sample-000000.mp3` | be careful with your prognostications said the stranger                                 | 1        | 0          |           |        |         | 00:05    | BE CAREFUL WITH YOUR PROGNOSTICATIONS SAID THE STRANGER                             |
| `cv-valid-dev/sample-000001.mp3` | then why should they be surprised when they see one                                     | 2        | 0          |           |        |         | 00:03    | THEN WHY SHOULD THEY BE SURPRISED WHEN THEY SEE ONE                                 |
| `cv-valid-dev/sample-000002.mp3` | a young arab also loaded down with baggage entered and greeted the englishman           | 2        | 0          |           |        |         | 00:05    | A YOUNG ARAB ALSO LOADED DOWN WITH BAGGAGE ENTERED AND GREETED THE ENGLISHMAN       |
| `cv-valid-dev/sample-000003.mp3` | i thought that everything i owned would be destroyed                                    | 3        | 0          |           |        |         | 00:04    | I FELT THAT EVERYTHING I OWNED WOULD BE DESTROYED                                   |
| `cv-valid-dev/sample-000004.mp3` | he moved about invisible but everyone could hear him                                    | 1        | 0          | forties   | female | england | 00:04    | HE MOVED ABOUT INVISIBLE BUT EVERY ONE COULD HEAR HIM                               |
| ...                              | ...                                                                                     | ...      | ...        | ...       | ...    | ...     | ...      | ...                                                                                 |

