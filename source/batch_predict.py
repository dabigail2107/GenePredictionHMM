#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_predict.py — Multi-FASTA batch decoder for the splice-aware H(S)MM

Description
-----------
Reads a multi-FASTA input file and runs `predict.py` on each sequence.
For every sequence, the original DNA and predicted state letters are written
into a single output FASTA file (two records per sequence).

Usage example (single line)
---------------------------
python source/batch_predict.py source/predict.py results/model.json results/test_out.fasta results/test_predicted.fasta --start-from-first-atg --stop-mode last --min-bg-len 30 --beam 20 --site-penalty 0.8

Notes
-----
- Each sequence is written as:
  >ID|original
  ACGT...
  >ID|predicted_states
  SSSEEEII...
- The wrapper creates a temporary FASTA for each sequence, calls the predictor,
  and extracts letters from its stdout.
"""

import os, sys, argparse, subprocess, tempfile, re

def read_multifasta(path: str):
    """Yield (header, sequence) pairs from a multi-FASTA file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header, chunks = None, []
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).replace("\n", "").replace("\r", "").strip().upper()
                header = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks).replace("\n", "").replace("\r", "").strip().upper()

def run_predict_once(predict_py: str, model_json: str, seq_id: str, seq: str, base_args: list[str]) -> str:
    """
    Run prediction for one sequence using the provided predictor script.
    Returns the predicted state letters.
    """
    with tempfile.TemporaryDirectory() as td:
        fasta_path = os.path.join(td, "one.fa")
        with open(fasta_path, "w", encoding="utf-8") as f:
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

        cmd = ["python", predict_py, model_json, fasta_path] + base_args
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Prediction failed for {seq_id}:\n{e.output}") from e

        # Extract predicted letters block
        match = re.search(r">prediction_states_letters\s*\r?\n([A-Z\-]+)", out)
        if not match:
            match = re.search(r">prediction_states_letters\s*\r?\n([^\r\n]+)", out)
        if not match:
            raise RuntimeError(f"Could not extract letters for {seq_id}.\nOutput (truncated):\n{out[:800]}")
        return match.group(1).strip()

def ensure_dir(path: str):
    """Create parent directory if needed."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Batch H(S)MM predictions: multi-FASTA → FASTA (original + predicted)")
    parser.add_argument("predict_py", help="Path to predict.py (or predict_ephase.py)")
    parser.add_argument("model_json", help="Trained model JSON")
    parser.add_argument("input_fasta", help="Multi-FASTA input file")
    parser.add_argument("output_fasta", help="Output FASTA (will contain DNA + predicted states)")
    parser.add_argument("predict_args", nargs=argparse.REMAINDER,
                        help="Extra args passed directly to predict.py (e.g. --start-from-first-atg --stop-mode last ...)")
    args = parser.parse_args()

    ensure_dir(args.output_fasta)
    total, success = 0, 0

    with open(args.output_fasta, "w", encoding="utf-8") as fout:
        for seq_id, seq in read_multifasta(args.input_fasta):
            total += 1
            try:
                letters = run_predict_once(args.predict_py, args.model_json, seq_id, seq, args.predict_args)
            except Exception as e:
                print(f"[WARN] {seq_id}: {e}", file=sys.stderr)
                continue

            fout.write(f">{seq_id}|original\n")
            for i in range(0, len(seq), 60):
                fout.write(seq[i:i+60] + "\n")
            fout.write(f">{seq_id}|predicted_states\n")
            for i in range(0, len(letters), 60):
                fout.write(letters[i:i+60] + "\n")
            success += 1

    print(f"[OK] {success}/{total} sequences processed → {args.output_fasta}")

if __name__ == "__main__":
    main()
