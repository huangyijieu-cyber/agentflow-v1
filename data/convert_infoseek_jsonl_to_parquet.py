#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def convert(input_jsonl: Path, output_parquet: Path) -> None:
    rows = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)

            sample_id = str(sample.get("id", "")).strip()
            question = str(sample.get("question", "")).strip()
            result = str(sample.get("result", "")).strip()

            if not sample_id:
                raise ValueError(f"line {line_no}: missing id")
            if not question:
                raise ValueError(f"line {line_no}: missing question")
            if not result:
                raise ValueError(f"line {line_no}: missing result")

            original_extra = sample.get("extra_info", {})
            if not isinstance(original_extra, dict):
                original_extra = {}

            # Keep only the metadata needed by the current InfoSeek reward pilot.
            # reward_spec is stored as JSON text to keep the parquet schema stable;
            # rollout.py already accepts reward_spec as either dict or JSON string.
            extra_info = {
                "source_dataset": str(
                    original_extra.get("source_dataset", "InfoSeek.jsonl")
                ),
                "source_line": int(original_extra.get("source_line", line_no)),
                "reward_spec": json.dumps(
                    sample.get("reward_spec", {}),
                    ensure_ascii=False,
                ),
                "final_answer": json.dumps(
                    sample.get("final_answer", {}),
                    ensure_ascii=False,
                ),
            }

            rows.append(
                {
                    "id": sample_id,
                    "question": question,
                    "chain": "",
                    "result": result,
                    "source": "infoseek",
                    "extra_info": extra_info,
                }
            )

    if not rows:
        raise ValueError(f"No valid samples found in {input_jsonl}")

    df = pd.DataFrame(
        rows,
        columns=["id", "question", "chain", "result", "source", "extra_info"],
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, engine="pyarrow", index=False)

    print(f"Input : {input_jsonl}")
    print(f"Output: {output_parquet}")
    print(f"Rows  : {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst row:")
    print(df.iloc[0].to_dict())


def main():
    parser = argparse.ArgumentParser(
        description="Convert InfoSeek reward-pilot JSONL to AgentFlow parquet format."
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_parquet", type=Path)
    args = parser.parse_args()

    convert(args.input_jsonl, args.output_parquet)


if __name__ == "__main__":
    main()
