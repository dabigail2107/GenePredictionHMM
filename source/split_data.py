#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_data.py
-------------
Splits a FASTA file of pairs (gene DNA; annotation) into train/test sets (and optional validation).

Input (required format):
  >ENSG00000123456_
  ACTG...
  >ENSG00000123456
  SGEEEIID...

Outputs:
- --train-out : DNA + annotation pairs (for training)
- --test-out  : DNA only (for prediction)
- --test-truth: DNA + annotation pairs (ground truth for evaluation)
- --val-out / --val-truth (optional): same as test but for validation

Usage (1 line examples):

# Simple 80/20 split
python source/split_data.py results/trainingData_out.fasta --train-out results/train_out.fasta --test-out results/test_out.fasta --test-truth results/test_truth.fasta --ratio 0.8 --seed 42

# 80/10/10 split (train/val/test)
python source/split_data.py results/trainingData_out.fasta --train-out results/train_out.fasta --val-out results/val_out.fasta --val-truth results/val_truth.fasta --test-out results/test_out.fasta --test-truth results/test_truth.fasta --ratio 0.8 --val-ratio 0.1 --seed 42
"""

import argparse
import random
import sys
from typing import List, Tuple

# Expected state letters used in this project
ALPHABET = set("SGEXAMDLIHC-")

def is_gene_header(h: str) -> bool:
    """Header for DNA line: must start with >ENSG and end with _"""
    return h.startswith(">ENSG") and h.endswith("_")

def is_annot_header(h: str) -> bool:
    """Header for annotation line: must start with >ENSG but not end with _"""
    return h.startswith(">ENSG") and not h.endswith("_")

def load_pairs(path: str) -> List[Tuple[str, str, str]]:
    """Load (gene_id, dna, annotation) pairs from structured FASTA."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    pairs = []
    i = 0
    while i + 3 < len(lines):
        h1, s1, h2, s2 = lines[i:i+4]
        i += 4

        if not (is_gene_header(h1) and is_annot_header(h2)):
            print(f"[WARN] Skipping malformed block around line {i}", file=sys.stderr)
            continue

        gene_id = h2[1:]  # remove '>'
        if len(s1) != len(s2):
            print(f"[WARN] {gene_id}: DNA({len(s1)}) != annot({len(s2)})", file=sys.stderr)

        bad = set(s2) - ALPHABET
        if bad:
            print(f"[WARN] {gene_id}: invalid chars {sorted(bad)}", file=sys.stderr)

        pairs.append((gene_id, s1, s2))

    if not pairs:
        raise RuntimeError(f"No valid pairs found in {path}")
    return pairs

def split_pairs(pairs: List[Tuple[str,str,str]], ratio=0.8, val_ratio=0.0, seed=42):
    """Randomly split pairs into train, val (optional), and test."""
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(ratio * n)
    n_val   = int(val_ratio * n)
    n_test  = n - n_train - n_val

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train+n_val]
    test  = shuffled[n_train+n_val:]
    return train, val, test

def write_train(pairs, out_path: str):
    """Write training pairs (DNA + annotation)."""
    with open(out_path, "w", encoding="utf-8") as out:
        for gid, dna, ann in pairs:
            out.write(f">{gid}_\n{dna}\n>{gid}\n{ann}\n")

def write_dna_only(pairs, out_path: str):
    """Write DNA-only FASTA (for test or val prediction)."""
    with open(out_path, "w", encoding="utf-8") as out:
        for gid, dna, _ in pairs:
            out.write(f">{gid}_\n{dna}\n")

def write_truth(pairs, out_path: str):
    """Write ground truth (DNA + annotation)."""
    write_train(pairs, out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split FASTA pairs into train/test/(val).")
    ap.add_argument("trainingData_fasta", help="Input FASTA (DNA + annotation)")
    ap.add_argument("--train-out", required=True, help="Output training FASTA (DNA+annot)")
    ap.add_argument("--test-out", required=True, help="Output test FASTA (DNA only)")
    ap.add_argument("--test-truth", required=True, help="Output test FASTA (DNA+annot ground truth)")
    ap.add_argument("--ratio", type=float, default=0.8, help="Training ratio (default 0.8)")
    ap.add_argument("--val-ratio", type=float, default=0.0, help="Validation ratio (optional, default 0)")
    ap.add_argument("--val-out", help="Validation FASTA (DNA only, optional)")
    ap.add_argument("--val-truth", help="Validation FASTA (DNA+annot, optional)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    pairs = load_pairs(args.trainingData_fasta)
    train, val, test = split_pairs(pairs, ratio=args.ratio, val_ratio=args.val_ratio, seed=args.seed)

    write_train(train, args.train_out)
    write_dna_only(test, args.test_out)
    write_truth(test, args.test_truth)

    if args.val_ratio > 0:
        if not (args.val_out and args.val_truth):
            print("Error: validation ratio set but no val-out or val-truth specified.", file=sys.stderr)
            sys.exit(2)
        write_dna_only(val, args.val_out)
        write_truth(val, args.val_truth)

    print(f"[OK] Total={len(pairs)} | Train={len(train)} | Val={len(val)} | Test={len(test)}")
    print(f"[OK] Written:")
    print(f"  - {args.train_out}")
    if args.val_ratio > 0:
        print(f"  - {args.val_out}")
        print(f"  - {args.val_truth}")
    print(f"  - {args.test_out}")
    print(f"  - {args.test_truth}")
