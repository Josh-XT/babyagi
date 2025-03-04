import requests
import json
import os
from Commands import Commands
from Config import Config

CFG = Config()

class audio_text(Commands):
    def __init__(self):
        super().__init__()
        if CFG.HUGGINGFACE_API_KEY is not None:
            self.commands = {
                "Read Audio from File": self.read_audio_from_file,
                "Read Audio": self.read_audio
            }

    def read_audio_from_file(self, audio_path: str):
        audio_path = os.path.join(CFG.WORKING_DIRECTORY, audio_path)
        with open(audio_path, "rb") as audio_file:
            audio = audio_file.read()
        return self.read_audio(audio)

    def read_audio(self, audio):
        model = CFG.HUGGINGFACE_AUDIO_TO_TEXT_MODEL
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        api_token = CFG.HUGGINGFACE_API_KEY
        headers = {"Authorization": f"Bearer {api_token}"}

        if api_token is None:
            raise ValueError(
                "You need to set your Hugging Face API token in the config file."
            )

        response = requests.post(
            api_url,
            headers=headers,
            data=audio,
        )

        text = json.loads(response.content.decode("utf-8"))["text"]
        return "The audio says: " + text
