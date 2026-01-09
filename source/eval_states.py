#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_states.py — Evaluate predicted state letters vs. ground truth.

Overview
--------
This evaluator compares per-base state-letter sequences (e.g., SSSEEEII…)
between a ground-truth multi-FASTA and a predicted multi-FASTA. It is tolerant
to header naming and layout differences:
- For each ID, it looks for sequences labeled with "|predicted_states" or
  "|truth_states". If not found, it heuristically picks the longest sequence
  whose characters are mostly from the state alphabet (not A/C/G/T/N).
- If the same ID appears multiple times, the longest candidate is used.

Outputs
-------
- Console summary:
  - Global accuracy
  - Global exon vs. non-exon metrics (precision, recall, F1, IoU, accuracy)
  - Per-class metrics (precision/recall/F1/support)
- Files in --outdir:
  - per_class_metrics.csv
  - confusion_matrix.csv
  - summary.json (includes per-sequence summary)

Single-line usage
-----------------
python source/eval_states.py results/test_truth.fasta results/test_predicted.fasta --outdir results/eval

Notes
-----
- By default, only 'E' is considered exon in the binary exon-vs-nonexon score.
  You can change this with --exon-letters (e.g. "E" or "EEX" if needed).
- The evaluator truncates truth/pred pairs to the shortest length to avoid
  misalignment issues across inputs prepared by different tools.
"""

import os, sys, argparse, json, csv, re
from collections import defaultdict

# Default alphabets (adjustable via CLI for exon letters)
STATE_LETTERS = set(list("SGEIXAMDLHC-"))   # Supported state letters (baseline)
NONSTATE_NT   = set(list("ACGTN"))          # Nucleotides (used to detect DNA vs. states)

def read_multifasta_states(path: str):
    """
    Read a multi-FASTA and return: id -> {"original": DNA (optional), "states": best-state-string}
    Heuristics:
      - If header contains 'predicted_states' or 'truth_states' or 'states', it's a state candidate.
      - If sequence contains only A/C/G/T/N → treat as original DNA.
      - Otherwise, if ≥80% of chars are in STATE_LETTERS → candidate state string.
      - For each ID, keep the longest candidate.
    """
    def flush_record():
        nonlocal cur_id, cur_head, buf, data
        if cur_id is None: return
        seq = "".join(buf).replace(" ", "").replace("\r", "").replace("\n", "").strip()
        if not seq: return
        rec = data.setdefault(cur_id, {})
        head_low = cur_head.lower()
        # Prefer explicit labels
        if ("predicted_states" in head_low) or ("truth_states" in head_low) or ("states" in head_low):
            rec.setdefault("states_candidates", []).append(seq)
        else:
            # DNA vs. states heuristic
            if set(seq) <= NONSTATE_NT:
                rec["original"] = seq
            else:
                valid = sum(1 for c in seq if c in STATE_LETTERS)
                if valid >= max(1, int(0.8 * len(seq))):
                    rec.setdefault("states_candidates", []).append(seq)
                else:
                    rec.setdefault("other", []).append(seq)

    data, cur_id, cur_head, buf = {}, None, "", []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                flush_record()
                cur_head = line[1:].strip()
                cur_id = cur_head.split()[0].split("|")[0]  # ID before the first pipe
                buf = []
            else:
                buf.append(line.strip())
        flush_record()

    # Pick longest candidate as the definitive states string
    for sid, rec in data.items():
        cands = rec.get("states_candidates", [])
        if cands:
            rec["states"] = max(cands, key=len)
        rec.pop("states_candidates", None)
    return data

def align_pair(truth: str, pred: str):
    """Truncate both strings to the shortest length to avoid misalignment artifacts."""
    n = min(len(truth), len(pred))
    return truth[:n], pred[:n]

def compute_confusion(truth: str, pred: str, ignore_set=set(["-"])):
    """Build a confusion matrix over labels (excluding any in ignore_set)."""
    labels = sorted(list(set(list(truth + pred)) - ignore_set))
    idx = {c: i for i, c in enumerate(labels)}
    M = [[0] * len(labels) for _ in range(len(labels))]
    for t, p in zip(truth, pred):
        if t in ignore_set: 
            continue
        if (t not in idx) or (p not in idx):
            continue
        M[idx[t]][idx[p]] += 1
    return labels, M

def per_class_metrics(labels, M):
    """Return per-class metrics: (label, tp, fp, fn, precision, recall, f1, support)."""
    out = []
    for i, lab in enumerate(labels):
        tp = M[i][i]
        fp = sum(M[r][i] for r in range(len(labels)) if r != i)
        fn = sum(M[i][c] for c in range(len(labels)) if c != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sup  = sum(M[i])
        out.append((lab, tp, fp, fn, prec, rec, f1, sup))
    return out

def base_accuracy(truth: str, pred: str, ignore="-"):
    """Per-base accuracy ignoring positions where truth == ignore."""
    N = 0; ok = 0
    for t, p in zip(truth, pred):
        if t == ignore:
            continue
        N += 1
        ok += (t == p)
    return ok / N if N > 0 else 0.0

def to_mask_exon(seq_letters: str, exon_letters: set):
    """Map state letters to {1: exon, 0: non-exon} using provided exon letter set."""
    return [1 if c in exon_letters else 0 for c in seq_letters]

def prf_from_masks(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, iou=iou, accuracy=acc)

def write_csv(path, rows, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description="Evaluate predicted state letters vs. ground truth.")
    ap.add_argument("truth_fasta", help="Multi-FASTA containing ground-truth states (and possibly DNA).")
    ap.add_argument("pred_fasta",  help="Multi-FASTA containing predicted states (and possibly DNA).")
    ap.add_argument("--outdir", default="results/eval", help="Output directory for reports.")
    ap.add_argument("--exon-letters", default="E", help="Letters treated as exon for binary metrics (default: 'E').")
    ap.add_argument("--ignore-label", default="-", help="Truth label to ignore in accuracy/confusion (default: '-')")
    args = ap.parse_args()

    exon_letters = set(list(args.exon_letters))

    truth = read_multifasta_states(args.truth_fasta)
    pred  = read_multifasta_states(args.pred_fasta)

    common = sorted(set(truth.keys()) & set(pred.keys()))
    if not common:
        print("No common IDs between truth and pred.", file=sys.stderr)
        sys.exit(2)

    T_concat, P_concat = [], []
    per_seq_summary = []

    for sid in common:
        t_states = truth[sid].get("states")
        p_states = pred[sid].get("states")
        if not t_states or not p_states:
            print(f"[WARN] Missing states for {sid} (truth={bool(t_states)} pred={bool(p_states)}). Skipping.", file=sys.stderr)
            continue

        t, p = align_pair(t_states, p_states)
        acc = base_accuracy(t, p, ignore=args.ignore_label)
        # exon vs non-exon
        m_true = to_mask_exon(t, exon_letters)
        m_pred = to_mask_exon(p, exon_letters)
        ex_metrics = prf_from_masks(m_true, m_pred)

        per_seq_summary.append({
            "id": sid, "len": len(t),
            "accuracy": acc,
            "exon_precision": ex_metrics["precision"],
            "exon_recall": ex_metrics["recall"],
            "exon_f1": ex_metrics["f1"],
            "exon_iou": ex_metrics["iou"],
            "exon_accuracy": ex_metrics["accuracy"]
        })
        T_concat.append(t); P_concat.append(p)

    if not per_seq_summary:
        print("Nothing to evaluate (no valid pairs).", file=sys.stderr)
        sys.exit(3)

    T_all = "".join(T_concat)
    P_all = "".join(P_concat)

    # Global metrics
    acc_all = base_accuracy(T_all, P_all, ignore=args.ignore_label)
    labels, M = compute_confusion(T_all, P_all, ignore_set=set([args.ignore_label]))
    pcm = per_class_metrics(labels, M)
    exon_global = prf_from_masks(to_mask_exon(T_all, exon_letters), to_mask_exon(P_all, exon_letters))

    # Console summary
    print(f"[GLOBAL] n_positions={len(T_all)}  accuracy={acc_all:.4f}")
    print(f"[GLOBAL] exon_vs_nonexon: precision={exon_global['precision']:.4f}  recall={exon_global['recall']:.4f}  F1={exon_global['f1']:.4f}  IoU={exon_global['iou']:.4f}  accuracy={exon_global['accuracy']:.4f}")
    print("\n[Per-class] label, tp, fp, fn, precision, recall, f1, support")
    for lab, tp, fp, fn, pr, re, f1, sup in pcm:
        print(f"{lab}\t{tp}\t{fp}\t{fn}\t{pr:.4f}\t{re:.4f}\t{f1:.4f}\t{sup}")

    # Files
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Confusion matrix CSV
    cm_rows = [[""] + labels] + [[labels[i]] + M[i] for i in range(len(labels))]
    write_csv(os.path.join(outdir, "confusion_matrix.csv"), cm_rows, header=None)

    # Per-class CSV
    pc_rows = [(lab, tp, fp, fn, f"{pr:.6f}", f"{re:.6f}", f"{f1:.6f}", sup) for lab, tp, fp, fn, pr, re, f1, sup in pcm]
    write_csv(os.path.join(outdir, "per_class_metrics.csv"), pc_rows,
              header=["label", "tp", "fp", "fn", "precision", "recall", "f1", "support"])

    # Summary JSON
    summary = {
        "global": {
            "n_positions": len(T_all),
            "accuracy": acc_all,
            "exon_vs_nonexon": exon_global,
            "labels": labels
        },
        "per_sequence": per_seq_summary
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Reports written to {outdir}")
    print(f"- {os.path.join(outdir, 'summary.json')}")
    print(f"- {os.path.join(outdir, 'per_class_metrics.csv')}")
    print(f"- {os.path.join(outdir, 'confusion_matrix.csv')}")

if __name__ == "__main__":
    main()
