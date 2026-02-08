import os
import csv
import random
import wave
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types


# ===================== CONFIG ===================== #

# Root dataset dir
# provide the path of the finala_data folder where the datasets are stored
BASE_DIR = "/media/RELETECH/final_data"

# How many total samples
N = 6

# Output folder name
OUTPUT_NAME = "gemini_deepfake"

# Gemini API
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY")


# Models
MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts"
]


# Voices
VOICES = {
    "bright": {
        "m": ["Puck", "Zephyr", "Orus"],
        "f": ["Aoede", "Laomedeia", "Leda"]
    },
    "firm": {
        "m": ["Charon", "Alnilam", "Gacrux", "Rasalgethi"],
        "f": ["Kore", "Vindemiatrix", "Schedar"]
    },
    "calm": {
        "m": ["Algieba", "Algenib", "Iapetus"],
        "f": ["Despina", "Callirrhoe", "Sadaltager", "Achird"]
    },
    "deep": {
        "m": ["Autonoe", "Enceladus", "Zubenelgenubi"],
        "f": ["Erinome", "Sulafat"]
    },
    "expressive": {
        "m": ["Fenrir", "Umbriel"],
        "f": ["Pulcherrima", "Sadachbia"]
    }
}


DATASETS = [
    "deepfake_data_mozilla",
    "deepfake_data_news",
    "deepfake_data_sust"
]


# ===================== HELPERS ===================== #


def read_metadata_csv(path):
    samples = []

    with open(path, encoding="utf8") as f:

        # Read first line to detect delimiter
        first_line = f.readline()

        if "|" in first_line and "," not in first_line:
            delimiter = "|"
        elif "," in first_line and "|" not in first_line:
            delimiter = ","
        else:
            # Fallback: prefer pipe if both or unclear
            delimiter = "|"

        # Reset file pointer
        f.seek(0)

        reader = csv.reader(f, delimiter=delimiter)

        # Skip header
        next(reader, None)

        for row in reader:

            if len(row) < 2:
                continue

            fid = row[0].strip()
            text = row[1].strip()

            if fid and text:
                samples.append((fid, text))

    return samples



def read_metadata_txt(path):
    samples = []

    with open(path, encoding="utf8") as f:
        for line in f:
            fid, text = line.strip().split("|", 1)
            samples.append((fid, text))

    return samples


def load_metadata(dataset_path):

    csv_path = dataset_path / "metadata.csv"
    txt_path = dataset_path / "metadata.txt"

    if csv_path.exists():
        return read_metadata_csv(csv_path)

    if txt_path.exists():
        return read_metadata_txt(txt_path)

    raise FileNotFoundError(f"No metadata in {dataset_path}")


def save_wav(path, pcm, rate=24000):

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


# ===================== TTS ===================== #


class GeminiTTS:

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def synthesize(self, text, model, voice):

        response = self.client.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
            )
        )

        return response.candidates[0].content.parts[0].inline_data.data


# ===================== DISTRIBUTION ===================== #


def build_voice_cycle():

    combos = []

    for style, genders in VOICES.items():
        for gender, voices in genders.items():
            for v in voices:
                combos.append((style, gender, v))

    random.shuffle(combos)
    return combos


def build_model_cycle():

    models = MODELS.copy()
    random.shuffle(models)
    return models


# ===================== MAIN PIPELINE ===================== #


def generate_dataset():

    client = GeminiTTS(API_KEY)

    per_dataset = N // 3
    per_model = per_dataset // len(MODELS)

    print(f"Total samples: {N}")
    print(f"Per dataset: {per_dataset}")
    print(f"Per model: {per_model}")

    voice_cycle = build_voice_cycle()
    model_cycle = build_model_cycle()

    voice_idx = 0
    model_idx = 0

    for dataset in DATASETS:

        print(f"\nProcessing: {dataset}")

        ds_path = Path(BASE_DIR) / dataset
        meta = load_metadata(ds_path)

        random.shuffle(meta)

        selected = meta[:per_dataset]

        out_root = ds_path / OUTPUT_NAME
        out_root.mkdir(exist_ok=True)

        counter = 0

        for fid, text in selected:

            model = model_cycle[model_idx]
            model_idx = (model_idx + 1) % len(model_cycle)

            model_dir = out_root / model
            model_dir.mkdir(exist_ok=True)

            style, gender, voice = voice_cycle[voice_idx]
            voice_idx = (voice_idx + 1) % len(voice_cycle)

            fname = f"{fid}_{style}_{gender}.wav"
            out_file = model_dir / fname

            print(f"→ {out_file.name}")

            try:
                audio = client.synthesize(
                    text=text,
                    model=model,
                    voice=voice
                )

                save_wav(out_file, audio)

                counter += 1

            except Exception as e:
                print("ERROR:", e)

        print(f"Generated {counter} samples in {dataset}")


# ===================== RUN ===================== #

if __name__ == "__main__":

    random.seed(42)

    generate_dataset()
