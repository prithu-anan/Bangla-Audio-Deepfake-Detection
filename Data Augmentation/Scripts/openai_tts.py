from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key="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"
)
speech_file_path = Path(__file__).parent / "speech.wav"

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="sage",
    input="কত দিন সময় লাগবে, সেটা কি কেউ বলতে পারবে?",
    instructions="Speak in a standard Bangla tone.",
    response_format="wav"
) as response:
    response.stream_to_file(speech_file_path)