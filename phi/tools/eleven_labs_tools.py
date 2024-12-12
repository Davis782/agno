"""
pip install elevenlabs
"""

from typing import Optional
from os import getenv, makedirs, path
from phi.tools import Toolkit
from phi.utils.log import logger
from phi.agent import Agent
from uuid import uuid4

try:
    from elevenlabs import ElevenLabs  # type: ignore
except ImportError:
    raise ImportError("`elevenlabs` not installed. Please install using `pip install elevenlabs`")


class ElevenLabsTools(Toolkit):
    def __init__(
        self,
        api_key: Optional[str] = None,
        save_path: str = "audio_generations",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_64",
    ):
        super().__init__(name="elevenlabs")

        self.api_key = api_key or getenv("ELEVEN_LABS_API_KEY")
        if not self.api_key:
            logger.error("ELEVEN_LABS_API_KEY not set. Please set the ELEVEN_LABS_API_KEY environment variable.")

        self.save_path = save_path
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format

        if not path.exists(self.save_path):
            makedirs(self.save_path, exist_ok=True)

        self.register(self.generate_audio)

    def generate_audio(self, agent: Agent, prompt: str) -> str:
        """
        Use this function to generate audio from a text prompt.

        Args:
            prompt (str): Text to generate audio from.
        Returns:
            str: Return the path to the generated audio file.
        """
        client = ElevenLabs(api_key=self.api_key)

        try:
            audio_generator = client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id=self.model_id,
                text=prompt,
                output_format=self.output_format,
            )

            output_filename = f"{uuid4()}.mp3"
            output_path = path.join(self.save_path, output_filename)

            with open(output_path, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            return f"Error: {e}"
