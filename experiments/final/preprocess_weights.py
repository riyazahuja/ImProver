# preprocess_weights.py
from datasets import Dataset
from transformers import AutoTokenizer
import sys


MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)


def build_example(inst, out, weight, train_on_inputs=False):
    # simple alpaca-style prompt: just instruction -> output
    prompt = inst.strip()
    target = out.strip()

    # tokenize both
    p = tokenizer(prompt, add_special_tokens=False)
    t = tokenizer(target, add_special_tokens=False)

    # concat: [prompt][eos?][target][eos]
    # (Qwen2.5 uses <|endoftext|> as eos_token)
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    input_ids = p["input_ids"] + t["input_ids"] + eos
    attention_mask = [1] * len(input_ids)

    # labels: mask prompt tokens if train_on_inputs=False
    labels = [-100] * len(p["input_ids"]) + t["input_ids"] + eos

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "weight": float(weight),  # <- keep this!
    }


def read_jsonl(path):
    import json

    for line in open(path, "r"):
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def convert(jsonl_path, out_dir, train_on_inputs=False):
    rows = []
    for ex in read_jsonl(jsonl_path):
        rows.append(
            build_example(
                ex["instruction"],
                ex.get("output", ""),
                ex.get("weight", 1.0),
                train_on_inputs,
            )
        )
    ds = Dataset.from_list(rows)
    # (Optional) set padded dtype so HF doesn’t try to cast weights
    # ds = ds.with_format("torch", columns=["input_ids","attention_mask","labels","weight"])
    ds.save_to_disk(out_dir)  # loads fast and preserves columns
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_weights.py <input_jsonl> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
    # names = [
    #     f"wSFT_{t}_{s}" for t in ["mark", "join", "replace"] for s in ["big", "small"]
    # ]
    # pairs = [
    #     (
    #         f"/home/riyaza/eval_improver/improver/experiments/results/length/replay_buffers/{name}.jsonl",
    #         f"/home/riyaza/eval_improver/improver/experiments/results/length/replay_buffers/{name}",
    #     )
    #     for name in names
    # ]
    # for pair in pairs:
    #     convert(
    #         pair[0],
    #         pair[1],
    #     )
    # # convert("/home/riyaza/eval_improver/improver/train/wST_test_data2.jsonl", "/home/riyaza/eval_improver/improver/train/prepped/wST_test_data2")
