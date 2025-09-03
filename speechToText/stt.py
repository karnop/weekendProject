import json
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()
# Get key from env
api_key = os.getenv("DEEPGRAM_API_KEY")

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

# Path to the audio file
AUDIO_FILE = "data/audio2.wav"

def sttHelper():
    try:
        deepgram = DeepgramClient(api_key=api_key)

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        #STEP 2: Configure Deepgram options for a
        # udio analysis
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options, timeout=300)

        res = response.to_json(indent=4)

        # STEP 4: Print the response
        with open("data/stt.json", 'w') as json_file:
            json_file.write(res)
        

    except Exception as e:
        print(f"Exception: {e}")


sttHelper()