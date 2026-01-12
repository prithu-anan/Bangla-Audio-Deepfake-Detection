from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os
import wave
import struct
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

voices = [
    # "CwhRBWXzGAHq8TQ4Fs17",
    # "EXAVITQu4vr4xnSDxMaL",
    # "FGY2WhTYpPnrIDTdsKH5",
    # "IKne3meq5aSn9XLyUdCD",
    # "JBFqnCBsd6RMkjVDRZzb",
    # "N2lVS1w4EtoT3dr4eOWO",
    # "SAz9YHcvj6GT2YYXdXww",
    # "SOYHLrjzK2X1ezoPC6cr",
    # "TX3LPaxmHKxFdv7VOQHJ",
    # "Xb7hH8MSUJpSbSDYk0k2",
    # "XrExE9yKIg1WjnnlVkGX",
    # "bIHbv24MWmeRgasZH58o",
    # "cgSgspJ2msm6clMCkdW9",
    # "cjVigY5qzO86Huf0OWal",
    # "iP95p4xoKVk53GoZ742B",
    # "nPczCjzI2devNBz1zQrb",
    # "onwK4e9ZLuTAKqWW03F9",
    # "pFZP5JQG7iQjIQuC4Bku",
    # "pNInz6obpgDQGcFmaJgB",
    "pqHfZKP75CvOlQylNhV4"
]

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

for voice_id in voices:
    audio_stream = client.text_to_speech.convert(
        text="কত দিন সময় লাগবে, সেটা কি কেউ বলতে পারবে?",
        voice_id=voice_id,
        model_id="eleven_v3",
        output_format="mp3_44100_128",
    )

    mp3_bytes = b"".join(audio_stream)

    with open(f"samples/voice_{voice_id}.mp3", "wb") as f:
        f.write(mp3_bytes)
