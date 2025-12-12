import argparse
import sys
from pathlib import Path
import pandas as pd
import json

def convert_json_to_parquet(inp: Path) -> int:
    if not inp.exists():
        print(f"输入文件不存在: {inp}", file=sys.stderr)
        return 1

    ext = inp.suffix.lower()
    try:
        if ext in ('.jsonl', '.ndjson'):
            df = pd.read_json(inp, lines=True)
        elif ext == '.json':
            # 优先尝试标准 JSON，再回退到 JSON Lines
            try:
                df = pd.read_json(inp, lines=False)
            except ValueError:
                df = pd.read_json(inp, lines=True)
        else:
            # 非标准后缀：尝试读取为 JSON 或 做 json_normalize
            try:
                df = pd.read_json(inp)
            except Exception:
                with open(inp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.json_normalize(data)
    except Exception as e:
        print(f"读取 JSON 失败: {e}", file=sys.stderr)
        return 2

    out = inp.with_suffix('.parquet')
    try:
        # 尝试默认引擎，若失败可指定 pyarrow/fastparquet
        try:
            df.to_parquet(out, index=False)
        except Exception:
            df.to_parquet(out, index=False, engine='pyarrow')
    except Exception as e:
        print(f"写入 Parquet 失败: {e}", file=sys.stderr)
        return 3

    print(out)
    return 0

def main():
    p = argparse.ArgumentParser(description="Convert JSON/JSONL to Parquet (save next to source).")
    p.add_argument("file", type=Path, help="输入 JSON/JSONL 文件路径")
    args = p.parse_args()
    sys.exit(convert_json_to_parquet(args.file))

if __name__ == "__main__":
    main()

"""
用法示例:
python data_converter.py /path/to/data.json
"""