#!/usr/bin/env bash

set -euo pipefail

# Merge LoRA adapter weights into a full model checkpoint.
#
# Usage:
#   bash merge_lora.sh --base /path/to/base_model --adapter /path/to/adapter --output /path/to/merged
#
# Optional:
#   --model-class causal|vision2seq   (default: causal)
#
# Environment variable alternatives:
#   BASE_MODEL, LORA_ADAPTER, MERGED_OUTPUT, MODEL_CLASS

BASE_MODEL="${BASE_MODEL:-}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
MERGED_OUTPUT="${MERGED_OUTPUT:-}"
MODEL_CLASS="${MODEL_CLASS:-vision2seq}"

print_help() {
  cat << 'EOF'
merge_lora.sh

Required arguments:
  --base PATH         Base model path (for example: /models/Qwen3-VL-8B-Instruct)
  --adapter PATH      LoRA adapter path (must contain adapter_config.json)
  --output PATH       Output path for merged checkpoint

Optional arguments:
  --model-class TYPE  Model loader type: causal or vision2seq (default: causal)
  -h, --help          Show this help

Examples:
  bash merge_lora.sh \
    --base /models/Qwen3-VL-8B-Instruct \
    --adapter /data00/checkpoints/my_adapter \
    --output /data00/checkpoints/qwen3vl_merged

  MODEL_CLASS=vision2seq BASE_MODEL=/models/Qwen3-VL-8B-Instruct \
  LORA_ADAPTER=/data00/checkpoints/my_adapter \
  MERGED_OUTPUT=/data00/checkpoints/qwen3vl_merged \
  bash merge_lora.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_MODEL="$2"
      shift 2
      ;;
    --adapter)
      LORA_ADAPTER="$2"
      shift 2
      ;;
    --output)
      MERGED_OUTPUT="$2"
      shift 2
      ;;
    --model-class)
      MODEL_CLASS="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if [[ -z "${BASE_MODEL}" || -z "${LORA_ADAPTER}" || -z "${MERGED_OUTPUT}" ]]; then
  echo "Error: --base, --adapter, --output are required." >&2
  print_help
  exit 1
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Error: base model path does not exist: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -d "${LORA_ADAPTER}" ]]; then
  echo "Error: adapter path does not exist: ${LORA_ADAPTER}" >&2
  exit 1
fi

if [[ ! -f "${LORA_ADAPTER}/adapter_config.json" ]]; then
  echo "Error: adapter_config.json not found in adapter path: ${LORA_ADAPTER}" >&2
  exit 1
fi

mkdir -p "${MERGED_OUTPUT}"

echo "[merge_lora] base model:   ${BASE_MODEL}"
echo "[merge_lora] adapter path: ${LORA_ADAPTER}"
echo "[merge_lora] output path:  ${MERGED_OUTPUT}"
echo "[merge_lora] model class:  ${MODEL_CLASS}"

python - << 'PY'
import os
import sys

from transformers import AutoTokenizer
from peft import PeftModel

base_model = os.environ["BASE_MODEL"]
adapter_path = os.environ["LORA_ADAPTER"]
output_path = os.environ["MERGED_OUTPUT"]
model_class = os.environ.get("MODEL_CLASS", "causal").lower()

if model_class == "causal":
    from transformers import AutoModelForCausalLM as ModelLoader
elif model_class == "vision2seq":
    from transformers import AutoModelForVision2Seq as ModelLoader
else:
    raise ValueError(f"Unsupported model class: {model_class}. Use causal or vision2seq.")

print(f"[merge_lora] loading base model with {ModelLoader.__name__} ...")
model = ModelLoader.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

print("[merge_lora] loading adapter ...")
model = PeftModel.from_pretrained(model, adapter_path)

print("[merge_lora] merging adapter into base model ...")
model = model.merge_and_unload()

print("[merge_lora] saving merged model ...")
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

print(f"[merge_lora] done. merged checkpoint saved to: {output_path}")
PY


# bash merge_lora.sh --base /models/Qwen3-VL-8B-Instruct --adapter /你的adapter目录 --output /你的merged输出目录 --model-class vision2seq