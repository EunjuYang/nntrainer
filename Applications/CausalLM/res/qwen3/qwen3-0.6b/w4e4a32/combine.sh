#!/bin/bash
# Combine split parts into the full quantized bin file.
# Usage: ./combine.sh [output_dir]
#
# If output_dir is omitted the combined file is written
# into the same directory as this script.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR}"

OUT_FILE="$OUT_DIR/nntr_qwen3_0.6b_w4e4a32.bin"
EXPECTED_SHA256="e677de3c6be6d6e5a4faa21adbf4adc52d731d1f9f028ef4eea09513e23f21cd"

echo "Combining parts -> $OUT_FILE"
cat "$SCRIPT_DIR"/nntr_qwen3_0.6b_w4e4a32.bin.part_* > "$OUT_FILE"

ACTUAL=$(sha256sum "$OUT_FILE" | awk '{print $1}')
if [ "$ACTUAL" = "$EXPECTED_SHA256" ]; then
  echo "OK  sha256 verified ($ACTUAL)"
else
  echo "FAIL  expected $EXPECTED_SHA256"
  echo "      got      $ACTUAL"
  exit 1
fi
