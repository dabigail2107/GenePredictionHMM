#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py — Splice-aware H(S)MM decoder (semi-Markov with geometric lengths)

Highlights
----------
- Terminal state: **cstop only**.
- Motif states (start/stop/splice) use k-mer emission tables from model["emission_probs"]["motifs"].
- Background states (exon/intron/exintron) use a simple, robust unigram backoff for emissions
  (fast and stable) and geometric length distributions from model["lengths"].
- Beam pruning controls speed vs. search breadth.
- Outputs: state letters, GFF3, and optional BED.

Input flexibility
-----------------
- If the second positional argument is a FASTA path, the first sequence in the file is decoded
  (for batch decoding of multi-FASTA, use your `predict_all.py` companion).
- If it’s a raw DNA string (ACGT), it will be decoded directly.

Example (single line)
---------------------
python source/predict.py model.json results/test_out.fasta --start-from-first-atg --stop-mode last --min-bg-len 30 --beam 10 --site-penalty 0.6 --letters results/test_letters.txt --gff3 results/test_predictions.gff3 --bed results/test_predictions.bed
"""

import os, sys, json, math, argparse, re
from functools import lru_cache
import numpy as np

EPS = 1e-12
NEG_INF = -1e18

# Background and motif state sets
BACKGROUND_STATES = {"exon", "intron", "exintron"}
MOTIF_STATES = {"cstart","cstartA","accsite","accAlt","donsite","donAlt","cstop","cstopA"}

# Rendering map for letters output
LETTER_MAP = {
    "cstart": "S", "cstartA": "G",
    "exon": "E", "intron": "I", "exintron": "X",
    "accsite": "A", "accAlt": "M",
    "donsite": "D", "donAlt": "L",
    "cstop": "H", "cstopA": "C"
}

def logp(x: float) -> float:
    """Safe log in probability space."""
    return NEG_INF if x <= 0.0 else math.log(x)

def read_sequence_and_id(arg: str, forced_id: str | None = None):
    """
    If `arg` is a file path, read the first FASTA record.
    Otherwise, treat `arg` as a raw DNA string.
    Returns (sequence, seq_id).
    """
    seq_id = forced_id or "seq1"
    if os.path.isfile(arg):
        header, seq_chunks = None, []
        with open(arg, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(">"):
                    if header is None:
                        header = line[1:].strip().split()[0]
                    continue
                seq_chunks.append(line.strip())
        s = re.sub(r"\s+", "", "".join(seq_chunks)).upper()
        if not s:
            raise RuntimeError("No sequence found in FASTA.")
        if not forced_id and header:
            seq_id = header
        return s, seq_id
    # Raw DNA string
    return arg.strip().upper(), seq_id

def find_first_atg(seq: str) -> int | None:
    m = re.search("ATG", seq)
    return m.start() if m else None

# ---------- Background emissions (fast unigram backoff) ----------
def _unigram_logmap(unigram: dict[str, float]) -> dict[str, float]:
    um = {b: float(unigram.get(b, 0.25)) for b in "ACGT"}
    s = sum(um.values()) or 1.0
    return {b: logp(max(EPS, um[b] / s)) for b in "ACGT"}

def bg_per_base_logs(seq: str, unigram: dict[str, float]) -> np.ndarray:
    umap = _unigram_logmap(unigram)
    out = np.empty(len(seq), dtype=np.float64)
    for i, c in enumerate(seq):
        out[i] = umap.get(c, logp(EPS))
    return out

# ---------- Motif emissions (precompute pos->logp) ----------
def build_motif_tables(emiss: dict, seq: str):
    n = len(seq)
    motifs = emiss.get("motifs", {}) or {}
    motif_k: dict[str, int] = {}
    motif_log: dict[str, np.ndarray] = {}
    for st, info in motifs.items():
        k = int(info.get("k", 3))
        prior = info.get("prior", {}) or {}
        arr = np.full(n + 1, NEG_INF, dtype=np.float64)
        if k <= n:
            p_else = float(prior.get("*", 0.0))
            for pos in range(k, n + 1):
                chunk = seq[pos - k:pos]
                p = float(prior.get(chunk, p_else))
                arr[pos] = logp(p if p > 0 else EPS)
        motif_k[st] = k
        motif_log[st] = arr
    return motif_k, motif_log

# ---------- Geometric lengths ----------
def geom_logpmf(L: int, mu: float) -> float:
    if mu <= 1e-9:
        return NEG_INF
    p = 1.0 / mu
    if p <= 0.0 or p >= 1.0:
        return NEG_INF
    return logp(p) + (L - 1) * logp(1.0 - p)

def precompute_geom(lengths_cfg: dict, max_len_cap: int, min_len_fallback: int):
    """
    Build (bounds, pmf) for each background state from model["lengths"].
    Falls back to min_len_fallback / max_len_cap if missing.
    """
    bounds: dict[str, tuple[int, int]] = {}
    pmf: dict[str, np.ndarray] = {}
    for st, info in lengths_cfg.items():
        mn = int(info.get("min", min_len_fallback))
        mx = int(info.get("max", max_len_cap))
        mu = float(info.get("mu0", 100.0))
        mn = max(1, mn)
        mx = max(mn, mx)
        bounds[st] = (mn, mx)
        arr = np.full(mx + 1, NEG_INF, dtype=np.float64)
        for L in range(mn, mx + 1):
            arr[L] = geom_logpmf(L, mu)
        pmf[st] = arr
    return bounds, pmf

# ---------- Rendering ----------
def to_letters(segments: list[tuple[str,int,int]], n: int) -> str:
    arr = ["-"] * n
    for st, i, j in segments:
        ch = LETTER_MAP.get(st, "?")
        for k in range(i, j):
            arr[k] = ch
    return "".join(arr)

def to_gff3(seq_id: str, segments: list[tuple[str,int,int]]) -> str:
    out = ["##gff-version 3"]
    for st, i, j in segments:
        out.append(f"{seq_id}\tpredict\t{st}\t{i+1}\t{j}\t.\t+\t.\tID={st}_{i}_{j};Name={st}")
    return "\n".join(out) + "\n"

def to_bed(seq_id: str, segments: list[tuple[str,int,int]]) -> str:
    return "\n".join(f"{seq_id}\t{i}\t{j}\t{st}\t0\t+" for st, i, j in segments) + "\n"

# ---------- Viterbi (pure Python, robust) ----------
def viterbi_python(
    seq: str,
    states: list[str],
    trans: dict,
    pi: dict,
    emiss: dict,
    lengths_cfg: dict,
    site_penalty: float = 0.0,
    stop_mode: str = "last",
    tail_allow: int = 300,
    beam: float = 6.0,
):
    """
    Semi-Markov Viterbi:
    - Background states use geometric length prior + per-base unigram backoff emissions.
    - Motif states are fixed-length segments (k-mers) using provided prior tables.
    - Beam pruning applied at each position for efficiency.

    Returns a list of segments: [(state, start, end), ...]
    """
    n = len(seq)
    S = len(states)
    sid = {s: i for i, s in enumerate(states)}

    # DP arrays
    dp = [[NEG_INF] * (n + 1) for _ in range(S)]
    bps = [[-1] * (n + 1) for _ in range(S)]  # back state index
    bpp = [[-1] * (n + 1) for _ in range(S)]  # back pos

    # Incoming transitions (t -> ns) with log weights
    ins: dict[str, list[tuple[str, float]]] = {s: [] for s in states}
    for a in states:
        row = trans.get(a, {}) or {}
        for b, w in row.items():
            if float(w) > 0.0:
                ins[b].append((a, logp(float(w))))

    # Background unigram prior (shared backoff)
    bg_uni = emiss.get("background", {}).get("prior", {}).get(
        "unigram", {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    )
    bg_logs = {st: bg_per_base_logs(seq, bg_uni) for st in BACKGROUND_STATES}

    # Motif tables (pos -> logp)
    motif_k, motif_log = build_motif_tables(emiss, seq)

    # Lengths (geometric)
    bounds, pmf = precompute_geom(lengths_cfg, n, 1)
    Lmin = {s: bounds.get(s, (1, 1))[0] for s in states}
    Lmax = {s: bounds.get(s, (1, 1))[1] for s in states}

    # Initialization at pos=0
    for s in states:
        i = sid[s]
        if s in BACKGROUND_STATES:
            mn, mx = Lmin[s], min(Lmax[s], n)
            lpi = logp(float(pi.get(s, 0.0)))
            if lpi <= NEG_INF / 2:
                continue
            for L in range(mn, mx + 1):
                e = float(np.sum(bg_logs[s][:L]))
                lpL = pmf[s][L] if s in pmf and L < len(pmf[s]) else NEG_INF
                sc = lpi + e + lpL
                if sc > dp[i][L]:
                    dp[i][L] = sc
                    bps[i][L] = S  # <START>
                    bpp[i][L] = 0
        elif s in MOTIF_STATES:
            k = motif_k.get(s, 3)
            if 1 <= k <= n:
                e = motif_log[s][k]
                if e > NEG_INF / 2:
                    sc = logp(float(pi.get(s, 0.0))) + e - site_penalty
                    if sc > dp[i][k]:
                        dp[i][k] = sc
                        bps[i][k] = S
                        bpp[i][k] = 0

    # Forward recursion over end positions (1..n)
    for pos in range(1, n + 1):
        best_here = max(dp[i][pos] for i in range(S))
        cutoff = best_here - float(beam)

        for ns in states:
            j = sid[ns]

            # Background destination (segment of length L >= 1)
            if ns in BACKGROUND_STATES:
                mn, mx = Lmin[ns], Lmax[ns]
                maxL = min(mx, n - pos)
                if maxL <= 0:
                    continue

                # Select best incoming predecessor at 'pos'
                best_prev, pred = NEG_INF, -1
                for (t, a_log) in ins[ns]:
                    it = sid[t]
                    prev = dp[it][pos]
                    if prev <= cutoff:
                        continue
                    val = prev + a_log
                    if val > best_prev:
                        best_prev, pred = val, it
                if pred == -1:
                    continue

                base = best_prev
                bl = bg_logs[ns]
                for L in range(mn, maxL + 1):
                    end = pos + L
                    e = float(np.sum(bl[pos:end]))
                    lpL = pmf[ns][L] if ns in pmf and L < len(pmf[ns]) else NEG_INF
                    sc = base + e + lpL
                    if sc > dp[j][end]:
                        dp[j][end] = sc
                        bps[j][end] = pred
                        bpp[j][end] = pos

            # Motif destination (fixed length k)
            elif ns in MOTIF_STATES:
                k = motif_k.get(ns, 3)
                end = pos + k
                if end > n:
                    continue

                # Best incoming predecessor at 'pos'
                best_prev, pred = NEG_INF, -1
                for (t, a_log) in ins[ns]:
                    it = sid[t]
                    prev = dp[it][pos]
                    if prev <= cutoff:
                        continue
                    val = prev + a_log
                    if val > best_prev:
                        best_prev, pred = val, it
                if pred == -1:
                    continue

                e = motif_log[ns][end]
                if e > NEG_INF / 2:
                    sc = best_prev + e - site_penalty
                    if sc > dp[j][end]:
                        dp[j][end] = sc
                        bps[j][end] = pred
                        bpp[j][end] = pos

    # Termination: **cstop only**
    allowed_final = ["cstop"]
    best_s, best_pos, best_score = None, None, NEG_INF

    def consider(pos: int):
        nonlocal best_s, best_pos, best_score
        for s in allowed_final:
            if s not in sid:
                continue
            i = sid[s]
            sc = dp[i][pos]
            if sc > best_score:
                best_score, best_s, best_pos = sc, i, pos

    if stop_mode == "end":
        consider(n)
    elif stop_mode == "tail":
        lo = max(0, n - tail_allow)
        for pos in range(lo, n + 1):
            consider(pos)
    elif stop_mode == "any":
        for pos in range(1, n + 1):
            consider(pos)
    else:  # "last"
        for pos in range(n, 0, -1):
            consider(pos)
            if best_s is not None:
                break

    # Fallback: if no terminal cstop reached, return the globally best (state,pos)
    if best_s is None:
        for i in range(S):
            for pos in range(1, n + 1):
                sc = dp[i][pos]
                if sc > best_score:
                    best_score, best_s, best_pos = sc, i, pos
        if best_s is None:
            raise RuntimeError("No feasible path found.")

    # Backtrace
    segments: list[tuple[str,int,int]] = []
    i, pos = best_s, best_pos
    START = S
    while True:
        pi_s = bps[i][pos]
        pi_p = bpp[i][pos]
        st = states[i]
        if pi_s == -1 or pi_p == -1:
            break
        if pi_s == START:
            segments.append((st, 0, pos))
            break
        segments.append((st, pi_p, pos))
        i, pos = pi_s, pi_p
    segments.reverse()
    return segments

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Splice-aware H(S)MM predictor (semi-Markov, geometric lengths)")
    ap.add_argument("model_json", help="Path to model.json")
    ap.add_argument("seq_or_fasta", help="Raw DNA string or FASTA file (first record used)")
    ap.add_argument("--start-from-first-atg", action="store_true", help="Trim sequence before first ATG")
    ap.add_argument("--stop-mode", choices=["end", "tail", "any", "last"], default="last",
                    help="Termination strategy (default: last)")
    ap.add_argument("--tail-allow", type=int, default=300, help="Tail window for stop-mode=tail (nt)")
    ap.add_argument("--max-len", type=int, default=5000, help="Global max length cap if not provided per state")
    ap.add_argument("--min-bg-len", type=int, default=30, help="Global min length for background states")
    ap.add_argument("--beam", type=float, default=6.0, help="Beam pruning width in log-space (higher = slower/more thorough)")
    ap.add_argument("--site-penalty", type=float, default=0.6, help="Penalty per motif segment (start/stop/splice)")
    ap.add_argument("--letters", help="Write predicted state letters to this file")
    ap.add_argument("--gff3", help="Write GFF3 annotations to this file")
    ap.add_argument("--bed", help="Write BED annotations to this file")
    ap.add_argument("--seq-id", help="Override sequence identifier for outputs")
    args = ap.parse_args()

    # Load model
    with open(args.model_json, "r", encoding="utf-8") as f:
        model = json.load(f)

    states = model["states"]
    trans = model["transition_probs"]
    pi = model.get("start_probs", model.get("pi", {})) or {}
    emiss = model["emission_probs"]
    lengths_cfg = model["lengths"]

    # Read sequence
    seq, seq_id = read_sequence_and_id(args.seq_or_fasta, args.seq_id)
    print(f"# seq_id={seq_id} length={len(seq)}")

    if args.start_from_first_atg:
        idx = find_first_atg(seq)
        if idx is None:
            raise RuntimeError("No ATG found; cannot --start-from-first-atg.")
        if idx > 0:
            seq = seq[idx:]
            print(f"# start-from-first-atg: start at {idx}, new length={len(seq)}")

    # Decode
    segments = viterbi_python(
        seq=seq,
        states=states,
        trans=trans,
        pi=pi,
        emiss=emiss,
        lengths_cfg=lengths_cfg,
        site_penalty=args.site_penalty,
        stop_mode=args.stop_mode,
        tail_allow=args.tail_allow,
        beam=args.beam,
    )

    # Render (note: prediction may end before full sequence, depending on stop-mode)
    pred_len = segments[-1][2] if segments else 0
    letters = to_letters(segments, pred_len)

    print(">prediction_states_letters")
    print(letters)
    print("\n# Segments (state\tstart\tend\tlength)")
    for st, i, j in segments:
        print(f"{st}\t{i}\t{j}\t{j-i}")

    # Outputs
    if args.letters:
        with open(args.letters, "w") as f:
            f.write(">prediction_states_letters\n")
            f.write(letters + "\n")
        print(f"[OK] Letters written: {args.letters}")
    if args.gff3:
        with open(args.gff3, "w") as f:
            f.write(to_gff3(seq_id, segments))
        print(f"[OK] GFF3 written: {args.gff3}")
    if args.bed:
        with open(args.bed, "w") as f:
            f.write(to_bed(seq_id, segments))
        print(f"[OK] BED written: {args.bed}")

if __name__ == "__main__":
    main()