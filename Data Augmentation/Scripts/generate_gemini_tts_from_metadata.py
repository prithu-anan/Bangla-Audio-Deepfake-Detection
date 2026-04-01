import argparse
import csv
import json
import os
import random
import re
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Gemini voice catalog grouped by style and gender (taken from gemini_tts.py).
VOICES = {
    "bright": {
        "m": ["Puck", "Zephyr", "Orus"],
        "f": ["Aoede", "Laomedeia", "Leda"],
    },
    "firm": {
        "m": ["Charon", "Alnilam", "Gacrux", "Rasalgethi"],
        "f": ["Kore", "Vindemiatrix", "Schedar"],
    },
    "calm": {
        "m": ["Algieba", "Algenib", "Iapetus"],
        "f": ["Despina", "Callirrhoe", "Sadaltager", "Achird"],
    },
    "deep": {
        "m": ["Autonoe", "Enceladus", "Zubenelgenubi"],
        "f": ["Erinome", "Sulafat"],
    },
    "expressive": {
        "m": ["Fenrir", "Umbriel"],
        "f": ["Pulcherrima", "Sadachbia"],
    },
}


DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_COUNT = 1000
DEFAULT_RATE_LIMIT_SLEEP = 35
RATE_LIMIT_BUFFER_SECONDS = 2


def flatten_voices() -> list[tuple[str, str, str]]:
    combos: list[tuple[str, str, str]] = []
    for style, genders in VOICES.items():
        for gender, names in genders.items():
            for name in names:
                combos.append((style, gender, name))
    return combos


def read_texts_from_metadata(metadata_csv: Path) -> list[str]:
    texts: list[str] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"'text' column not found in {metadata_csv}")

        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)

    return texts


def build_balanced_voice_schedule(
    total_samples: int, voices: list[tuple[str, str, str]], seed: int
) -> list[tuple[str, str, str]]:
    if not voices:
        raise ValueError("Voice list is empty.")

    rng = random.Random(seed)
    shuffled_voices = voices.copy()
    rng.shuffle(shuffled_voices)

    per_voice = total_samples // len(shuffled_voices)
    remainder = total_samples % len(shuffled_voices)

    schedule: list[tuple[str, str, str]] = []
    for i, voice_info in enumerate(shuffled_voices):
        take = per_voice + (1 if i < remainder else 0)
        schedule.extend([voice_info] * take)

    rng.shuffle(schedule)
    return schedule


def save_wav(path: Path, pcm_bytes: bytes, sample_rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def load_checkpoint(checkpoint_path: Path) -> set[int]:
    if not checkpoint_path.exists():
        return set()

    with checkpoint_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    indices = payload.get("successful_indices", [])
    if not isinstance(indices, list):
        return set()

    return {int(i) for i in indices if isinstance(i, int) or str(i).isdigit()}


def save_checkpoint(
    checkpoint_path: Path,
    successful_indices: set[int],
    metadata_csv: Path,
    output_dir: Path,
    model: str,
    sample_count: int,
    seed: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata_csv": str(metadata_csv),
        "output_dir": str(output_dir),
        "model": model,
        "sample_count": sample_count,
        "seed": seed,
        "successful_indices": sorted(successful_indices),
        "successful_count": len(successful_indices),
    }
    tmp_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(checkpoint_path)


def get_rate_limit_wait_seconds(exc: Exception) -> int | None:
    message = str(exc)
    upper_msg = message.upper()

    if (
        "RESOURCE_EXHAUSTED" not in upper_msg
        and "429" not in upper_msg
        and "RATE LIMIT" not in upper_msg
        and "QUOTA" not in upper_msg
    ):
        return None

    retry_info_match = re.search(r"retryDelay['\"]?:\s*['\"]([0-9]+(?:\.[0-9]+)?)s['\"]", message)
    if retry_info_match:
        wait_sec = float(retry_info_match.group(1))
        return int(wait_sec) + RATE_LIMIT_BUFFER_SECONDS

    message_match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if message_match:
        wait_sec = float(message_match.group(1))
        return int(wait_sec) + RATE_LIMIT_BUFFER_SECONDS

    return DEFAULT_RATE_LIMIT_SLEEP


class GeminiTTSClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def synthesize(self, text: str, voice_name: str) -> bytes:
        response = self.client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )
        return response.candidates[0].content.parts[0].inline_data.data


def generate(
    metadata_csv: Path,
    output_dir: Path,
    sample_count: int,
    model: str,
    retries: int,
    seed: int,
    checkpoint_path: Path,
) -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Set it in your environment or .env file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    texts = read_texts_from_metadata(metadata_csv)
    if len(texts) < sample_count:
        raise ValueError(
            f"Requested {sample_count} samples, but only {len(texts)} non-empty text rows found."
        )

    selected_texts = texts[:sample_count]
    voice_list = flatten_voices()
    voice_schedule = build_balanced_voice_schedule(sample_count, voice_list, seed)
    successful_indices = load_checkpoint(checkpoint_path)

    tts = GeminiTTSClient(api_key=api_key, model=model)

    print(f"Metadata: {metadata_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {model}")
    print(f"Total voices available: {len(voice_list)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Already completed: {len(successful_indices)}")
    print(f"Generating {sample_count} files...")

    generated = 0
    failed = 0
    skipped = 0

    for idx, text in enumerate(selected_texts):
        style, gender, voice_name = voice_schedule[idx]
        out_name = f"gemini_{idx:04d}_{style}_{gender}_{voice_name}.wav"
        out_path = output_dir / out_name

        if idx in successful_indices:
            if out_path.exists():
                skipped += 1
                print(f"[{idx + 1}/{sample_count}] SKIP {out_name}")
                continue
            # If checkpoint says done but file is missing, regenerate and fix checkpoint.
            successful_indices.remove(idx)
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                successful_indices=successful_indices,
                metadata_csv=metadata_csv,
                output_dir=output_dir,
                model=model,
                sample_count=sample_count,
                seed=seed,
            )

        success = False
        for attempt in range(1, retries + 2):
            try:
                pcm_audio = tts.synthesize(text=text, voice_name=voice_name)
                save_wav(out_path, pcm_audio)
                successful_indices.add(idx)
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    successful_indices=successful_indices,
                    metadata_csv=metadata_csv,
                    output_dir=output_dir,
                    model=model,
                    sample_count=sample_count,
                    seed=seed,
                )
                generated += 1
                success = True
                print(f"[{idx + 1}/{sample_count}] OK  {out_name}")
                break
            except Exception as exc:
                rate_limit_wait = get_rate_limit_wait_seconds(exc)

                if attempt > retries:
                    print(f"[{idx + 1}/{sample_count}] FAIL {out_name} :: {exc}")
                    failed += 1
                    if rate_limit_wait is not None:
                        print(
                            f"[{idx + 1}/{sample_count}] RATE LIMIT after failure, sleeping {rate_limit_wait}s"
                        )
                        time.sleep(rate_limit_wait)
                else:
                    backoff = min(8, 2 * attempt)
                    wait_seconds = rate_limit_wait if rate_limit_wait is not None else backoff
                    print(
                        f"[{idx + 1}/{sample_count}] RETRY {attempt}/{retries} {out_name} :: {exc}"
                    )
                    print(f"[{idx + 1}/{sample_count}] Sleeping {wait_seconds}s before retry...")
                    time.sleep(wait_seconds)

        if not success:
            continue

    print("\nDone.")
    print(f"Generated: {generated}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total completed in checkpoint: {len(successful_indices)}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent  # Data Augmentation/

    parser = argparse.ArgumentParser(
        description=(
            "Generate Gemini TTS WAV files from metadata.csv using only the 'text' column, "
            "with balanced voice-type distribution."
        )
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=base_dir / "metadata.csv",
        help="Path to metadata.csv containing at least a 'text' column.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=base_dir / "Data" / "gemini_deepfake",
        help="Directory where generated WAV files are stored.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Gemini TTS model name.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per sample after first failure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic voice scheduling.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base_dir / "Data" / "gemini_deepfake" / "tts_checkpoint.json",
        help="Checkpoint file used to resume completed conversions.",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be > 0")

    generate(
        metadata_csv=args.metadata_csv.resolve(),
        output_dir=args.output_dir.resolve(),
        sample_count=args.count,
        model=args.model,
        retries=args.retries,
        seed=args.seed,
        checkpoint_path=args.checkpoint.resolve(),
    )


if __name__ == "__main__":
    main()
