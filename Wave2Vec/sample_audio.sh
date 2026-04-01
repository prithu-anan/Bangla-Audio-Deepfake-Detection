#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash sample_audio.sh [SOURCE_DIR] [OUTPUT_DIR]
# Example:
#   bash sample_audio.sh final_data sampled_data

SOURCE_DIR="${1:-final_data}"
OUTPUT_DIR="${2:-sampled_data}"

REAL_TARGET=2400
FAKE_TARGET=1000

REAL_OUT="$OUTPUT_DIR/real_samples"
FAKE_OUT="$OUTPUT_DIR/fake_samples"

mkdir -p "$REAL_OUT" "$FAKE_OUT"

# Collect audio files (adjust extensions if needed)
mapfile -d '' REAL_FILES < <(
  find "$SOURCE_DIR" -type f -path "*/real_wav/*" \
    \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.ogg" \) -print0
)

mapfile -d '' FAKE_FILES < <(
  find "$SOURCE_DIR" -type f -path "*/deepfake_wav/*" \
    \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.ogg" \) -print0
)

REAL_COUNT="${#REAL_FILES[@]}"
FAKE_COUNT="${#FAKE_FILES[@]}"

if (( REAL_COUNT < REAL_TARGET )); then
  echo "Error: only $REAL_COUNT real files found, need $REAL_TARGET"
  exit 1
fi

if (( FAKE_COUNT < FAKE_TARGET )); then
  echo "Error: only $FAKE_COUNT fake files found, need $FAKE_TARGET"
  exit 1
fi

echo "Found $REAL_COUNT real files and $FAKE_COUNT fake files."
echo "Sampling $REAL_TARGET real + $FAKE_TARGET fake..."

# Pick random unique files and copy
printf '%s\0' "${REAL_FILES[@]}" | shuf -z -n "$REAL_TARGET" | while IFS= read -r -d '' f; do
  rel="${f#"$SOURCE_DIR"/}"
  safe_name="${rel//\//__}"   # avoid name collisions
  cp "$f" "$REAL_OUT/$safe_name"
done

printf '%s\0' "${FAKE_FILES[@]}" | shuf -z -n "$FAKE_TARGET" | while IFS= read -r -d '' f; do
  rel="${f#"$SOURCE_DIR"/}"
  safe_name="${rel//\//__}"
  cp "$f" "$FAKE_OUT/$safe_name"
done

echo "Done."
echo "Real samples copied to: $REAL_OUT"
echo "Fake samples copied to: $FAKE_OUT"