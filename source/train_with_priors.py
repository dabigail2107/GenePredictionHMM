#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_with_priors.py
-----------------------
Training version that re-estimates both:
  (1) transition probabilities between HMM states, and
  (2) background emissions (Markov model for exon/intron/exintron)
while preserving the topology and motif definitions from priors.json.

Motif priors (start/stop/splice sites) are updated using
Dirichlet smoothing. Optionally, length means (mu0) can be
re-estimated from observed segment lengths.

Usage:
  python source/train_with_priors.py results/train_out.fasta source/priors.json model.json --order 4 --min-bg-len 30 --update-mu0
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import product
from typing import Dict, List, Tuple

LETTER_TO_STATE = {
    'S': 'cstart',
    'G': 'cstartA',
    'E': 'exon',
    'X': 'exintron',
    'A': 'accsite',
    'M': 'accAlt',
    'D': 'donsite',
    'L': 'donAlt',
    'I': 'intron',
    'H': 'cstop',
    'C': 'cstopA',
    '-': 'intergenic'
}

BACKGROUND_STATES = {'exon', 'intron', 'exintron'}


# -------------------- Utility Functions --------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train model with re-estimated transitions and background emissions."
    )
    p.add_argument("train_fasta", help="Training FASTA with DNA + annotation pairs")
    p.add_argument("priors_json", help="Input priors.json (defines topology)")
    p.add_argument("out_model_json", help="Output model.json file")
    p.add_argument("--order", type=int, default=4, help="Markov order for background emissions")
    p.add_argument("--min-bg-len", type=int, default=10, help="Min segment length to keep for background")
    p.add_argument("--update-mu0", action="store_true", help="Update lengths.mu0 using posterior means")
    return p.parse_args()


def read_pairs(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i + 3 < len(lines):
        h1, seq, h2, ann = lines[i:i+4]
        i += 4
        if not (h1.startswith(">ENSG") and h1.endswith("_") and h2.startswith(">ENSG")):
            continue
        gid1 = h1[1:-1]; gid2 = h2[1:]
        if gid1 != gid2:
            continue
        items.append((gid1, seq.upper(), ann))
    return items


def segments_from_letters(letters: str):
    segs = []
    if not letters:
        return segs
    cur = letters[0]
    start = 0
    for i, c in enumerate(letters[1:], start=1):
        if c != cur:
            segs.append((cur, start, i))
            cur, start = c, i
    segs.append((cur, start, len(letters)))
    return segs


def normalize(d):
    s = sum(d.values()) or 1.0
    return {k: v/s for k, v in d.items()}


def posterior_mean_geom(lengths, mu0, gamma):
    if not lengths:
        return mu0
    return (gamma * mu0 + sum(lengths)) / (gamma + len(lengths))


# -------------------- Training core --------------------

def train_model(train_fasta, priors, order=4, min_bg_len=10, update_mu0=False):
    motif_cfg = priors["emission_probs"]["motifs"]
    bg_cfg = priors["emission_probs"]["background"]
    lengths_cfg = priors["lengths"]
    states = priors["states"]

    motif_counts = {st: Counter() for st in motif_cfg.keys()}
    bg_counts = {st: defaultdict(lambda: defaultdict(int)) for st in BACKGROUND_STATES}
    bg_lengths = {st: [] for st in BACKGROUND_STATES}
    trans_counts = defaultdict(lambda: defaultdict(int))

    # Read annotated training pairs
    items = read_pairs(train_fasta)

    for _, dna, ann in items:
        dna = re.sub(r"[^ACGT]", "N", dna)
        segL = segments_from_letters(ann)

        # Map annotation letters → HMM states
        segS = []
        for (ch, i, j) in segL:
            st = LETTER_TO_STATE.get(ch)
            if st and st != "intergenic":
                segS.append((st, i, j))

        # Transition counts
        if segS:
            trans_counts["<START>"][segS[0][0]] += 1
            for (a, _1, _2), (b, _3, _4) in zip(segS, segS[1:]):
                trans_counts[a][b] += 1

        # Length distributions + background emissions
        for st, i, j in segS:
            L = j - i
            if st in BACKGROUND_STATES and L >= min_bg_len:
                bg_lengths[st].append(L)
                seq = dna[i:j]
                if len(seq) > order:
                    for k in range(order, len(seq)):
                        hist = seq[k - order:k]
                        nxt = seq[k]
                        if re.fullmatch(r"[ACGT]{%d}" % (order + 1), hist + nxt):
                            bg_counts[st][hist][nxt] += 1

        # Motif emissions (short k-mers)
        for st, cfg in motif_cfg.items():
            k = int(cfg.get("k", 3))
            for (st2, i, j) in segS:
                if st2 == st and j - i >= k:
                    km = dna[i:i+k]
                    if re.fullmatch(r"[ACGT]{%d}" % k, km):
                        motif_counts[st][km] += 1

    # --- Transition probabilities ---
    trans_probs = {}
    for src, row in trans_counts.items():
        trans_probs[src] = normalize(row)

    # Keep missing states as empty rows
    for st in states:
        trans_probs.setdefault(st, {})

    # --- Motif emissions ---
    motif_post = {}
    for st, cfg in motif_cfg.items():
        k = int(cfg.get("k", 3))
        beta = float(cfg.get("beta", 0.0))
        prior = dict(cfg.get("prior", {}))
        all_kmers = [''.join(p) for p in product("ACGT", repeat=k)]

        # Expand prior
        base = {km: 0.0 for km in all_kmers}
        sump = 0.0
        for km, p in prior.items():
            if km != "*" and km in base:
                base[km] = float(p)
                sump += float(p)
        wildcard = float(prior.get("*", 0.0))
        if wildcard > 0.0:
            rem = [km for km in all_kmers if base[km] == 0.0]
            add = wildcard / max(1, len(rem))
            for km in rem:
                base[km] = add
            sump += wildcard

        if sump > 0.0:
            base = {km: (base[km] / sump) * beta for km in all_kmers}
        else:
            base = {km: beta / len(all_kmers) for km in all_kmers}

        obs = motif_counts.get(st, Counter())
        post = {km: base.get(km, 0.0) + obs.get(km, 0.0) for km in all_kmers}
        motif_post[st] = {"k": k, "beta": beta, "prior": normalize(post)}

    # --- Background emissions ---
    beta_bg = float(bg_cfg.get("beta", 0.0))
    unigram = dict(bg_cfg.get("prior", {}).get("unigram", {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}))
    s = sum(unigram.values()) or 1.0
    unigram = {b: v/s for b, v in unigram.items()}

    bg_model = {}
    for st in BACKGROUND_STATES:
        bg_model[st] = {}
        for hist, nxts in bg_counts[st].items():
            row = {}
            for b in "ACGT":
                row[b] = nxts.get(b, 0) + beta_bg * unigram.get(b, 0.25)
            bg_model[st][hist] = normalize(row)

    # --- Lengths (mu0 update optional) ---
    if update_mu0:
        for st in BACKGROUND_STATES:
            cfg = lengths_cfg.get(st, {})
            mu0 = float(cfg.get("mu0", 100.0))
            gamma = float(cfg.get("gamma", 10.0))
            Ls = bg_lengths.get(st, [])
            mu_post = posterior_mean_geom(Ls, mu0, gamma)
            minL = int(cfg.get("min", 1))
            maxL = int(cfg.get("max", 1e6))
            cfg["mu0"] = max(minL, min(maxL, mu_post))

    # --- Assemble model ---
    model = {
        "states": states,
        "start_probs": priors["start_probs"],  # unchanged
        "transition_probs": trans_probs,
        "emission_probs": {
            "motifs": motif_post,
            "background": {
                "order": order,
                "beta": beta_bg,
                "prior": {"unigram": unigram},
                "probs": bg_model
            }
        },
        "lengths": lengths_cfg
    }

    return model


# -------------------- Main --------------------

def main():
    args = parse_args()
    with open(args.priors_json, "r", encoding="utf-8") as f:
        priors = json.load(f)

    model = train_model(
        args.train_fasta,
        priors,
        order=args.order,
        min_bg_len=args.min_bg_len,
        update_mu0=args.update_mu0
    )

    with open(args.out_model_json, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"[OK] model written: {args.out_model_json}")
    print("[OK] transitions re-estimated from training data")
    print("[OK] background Markov emissions re-estimated")
    if args.update_mu0:
        print("[OK] mu0 updated from observed lengths")


if __name__ == "__main__":
    main()
