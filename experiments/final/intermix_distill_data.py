#!/usr/bin/env python3
"""Intermix raw training data with distillation data."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Combine raw training data with a fraction of distillation data"
    )
    parser.add_argument("--raw", required=True, help="Path to raw JSONL file")
    parser.add_argument("--distill", required=True, help="Path to distill JSONL file")
    parser.add_argument(
        "--distill_frac",
        type=float,
        required=True,
        help="Fraction of distill data to include (0.0 to 1.0)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    assert 0.0 <= args.distill_frac <= 1.0, "distill_frac must be between 0 and 1"

    # Read distill lines and compute how many to include
    with open(args.distill, "r") as f:
        distill_lines = f.readlines()
    num_distill = int(args.distill_frac * len(distill_lines))

    # Write raw + subset of distill to output
    with open(args.output, "w") as out:
        for line in distill_lines[:num_distill]:
            out.write(line)
        with open(args.raw, "r") as f:
            for line in f:
                out.write(line)


if __name__ == "__main__":
    main()
