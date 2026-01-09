#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
format_data.py — Build HMM-compatible annotations from gene FASTA + exon lists.

Inputs
------
1) target.fasta        : one record per gene (FASTA)

2) exonlist.txt        : blocks per transcript: '>ENST' then <tx_start> <tx_end> <gene_start> <gene_end>
Coordinates are given relative to the transcript and its corresponding gene sequence. 
Only the gene coordinates (3rd and 4th columns) are used by the formatter to build the HMM annotation.

3) source2target.txt   : two columns 'ENST ENSG' (sorted by gene)

4) states.json (opt.)  : model states + allowed transitions (defaults to 'states.json')

Outputs
-------
- annotations_out.fasta : for each gene, DNA + consensus + one line per transcript
- trainingData_out.fasta: paired (gene DNA; gene consensus) sequences for training

Letters / states:
  S=cstart, G=cstartA, E=exon, X=exintron, A=accsite, M=accAlt,
  D=donsite, L=donAlt, I=intron, H=cstop, C=cstopA, -=gap

Usage
-----
python source/format_data.py data/human_target.fasta data/human_initialsourceexonlist.txt data/human_initialsource2target.fasta results/annotations_out.fasta results/trainingData_out.fasta --states source/states.json

"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from typing import Dict, List, Tuple
from Bio import SeqIO

# ------------------- Constants -------------------
CHAR_TO_STATE = {
    'S':'cstart','G':'cstartA','E':'exon','X':'exintron',
    'A':'accsite','M':'accAlt','D':'donsite','L':'donAlt',
    'I':'intron','H':'cstop','C':'cstopA','-':'intergenic'
}
STATE_TO_CHAR = {v:k for k,v in CHAR_TO_STATE.items()}
BACKGROUND = {'E','I','X'}

# ------------------- CLI -------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Format gene+transcript inputs into HMM-letter annotations."
    )
    p.add_argument("target_fasta", help="FASTA with genes (one record per gene)")
    p.add_argument("exonlist", help="Transcript exon list (blocks: >ENST then lines 'chrom strand start end')")
    p.add_argument("source2target", help="Mapping file: 'ENST ENSG' per line, sorted by gene")
    p.add_argument("annotations_out", help="Output FASTA: DNA + consensus + one annotation per transcript")
    p.add_argument("training_out", help="Output FASTA: training pairs (gene DNA; gene consensus)")
    p.add_argument("--states", default="states.json", help="states.json with 'states' and 'transitions' (default: states.json)")
    p.add_argument("-v","--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

# ------------------- IO helpers -------------------
def read_genes(fasta_path: Path) -> List[Tuple[str,str]]:
    out = []
    with fasta_path.open() as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            out.append((rec.id, str(rec.seq).upper()))
    if not out:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return out

def read_exonlist(path: Path) -> List[List[str]]:
    blocks: List[List[str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                blocks.append([line.strip(">").strip()])
            else:
                toks = line.split()
                if len(toks) == 4:
                    blocks[-1].append(toks)
                else:
                    logging.warning("Skipping malformed exon line: %r", line)
    if not blocks:
        raise ValueError(f"No transcripts parsed from {path}")
    return blocks

def read_source2target(path: Path) -> List[Tuple[str,str]]:
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            tx, gn = line.split()[:2]
            pairs.append((tx, gn))
    if not pairs:
        raise ValueError(f"No transcript→gene pairs in {path}")
    return pairs

def read_states(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    with path.open(encoding="utf-8") as fh:
        H = json.load(fh)
    states = H["states"]
    transitions = H["transitions"]
    return states, transitions

# ------------------- Transitions -------------------
def build_char_transitions(states: List[str], transitions: Dict[str, List[str]]) -> Dict[str, set]:
    tchar: Dict[str,set] = {}
    for s in states:
        sc = STATE_TO_CHAR[s]
        tchar.setdefault(sc, set()).add(sc)  # self
        for nxt in transitions.get(s, []):
            tchar[sc].add(STATE_TO_CHAR[nxt])
    tchar.setdefault('-', set()).add('-')
    return tchar

def is_valid_transition_char(tchar: Dict[str,set], a: str, b: str) -> bool:
    return (a == b) or (b in tchar.get(a, set()))

# ------------------- Annotation builders -------------------
def annotate_transcript(glen: int, exons: List[List[str]]) -> str:
    """
    Build full-length HMM-letter string for a transcript over the gene span.
    Exon lines are [<tx_start> <tx_end> <gene_start> <gene_end>] as parsed.
    """
    ann = ['-'] * glen
    if not exons:
        return ''.join(ann)

    def clamp(i: int) -> bool:
        return 0 <= i < glen

    if len(exons) == 1:
        s = int(exons[0][2]); e = int(exons[0][3])
        if s <= glen-3:
            ann[s:s+3] = ['S','S','S']
        for i in range(s+3, max(s+3, e-3)):
            if clamp(i): ann[i] = 'E'
        if e-3 >= 0:
            ann[max(0,e-3):min(glen,e)] = ['H','H','H']
        return ''.join(ann)

    # first exon + initial donor
    s0 = int(exons[0][2]); e0 = int(exons[0][3])
    if s0 <= glen-3:
        ann[s0:s0+3] = ['S','S','S']
    for i in range(s0+3, e0):
        if clamp(i): ann[i] = 'E'
    if e0 < glen-1:
        ann[e0:e0+2] = ['D','D']

    prev_end = e0
    # internal exons
    for e in range(1, len(exons)-1):
        s = int(exons[e][2]); en = int(exons[e][3])
        for i in range(prev_end+2, s-2):
            if clamp(i): ann[i] = 'I'
        if s-2 >= 0:
            ann[max(0,s-2):s] = ['A','A']
        for i in range(s, en):
            if clamp(i): ann[i] = 'E'
        if en < glen-1:
            ann[en:en+2] = ['D','D']
        prev_end = en

    # last exon + terminal stop
    sl = int(exons[-1][2]); el = int(exons[-1][3])
    for i in range(prev_end+2, sl-2):
        if clamp(i): ann[i] = 'I'
    if sl-2 >= 0:
        ann[max(0,sl-2):sl] = ['A','A']
    for i in range(sl, max(sl, el-3)):
        if clamp(i): ann[i] = 'E'
    if el-3 >= 0:
        ann[max(0,el-3):min(glen,el)] = ['H','H','H']

    return ''.join(ann)

def per_position_sets(g_tracks: List[str]) -> List[set]:
    """Merge transcript letters per position into sets (ignore '#')"""
    dna = g_tracks[0]
    L = len(dna)
    pos_sets: List[set] = []
    for i in range(L):
        s = set()
        for t in g_tracks[1:]:
            if t and i < len(t):
                c = t[i]
                if c != '#':
                    s.add(c)
        pos_sets.append(s if s else {'-'})
    return pos_sets

def consensus_from_sets(sets_per_pos: List[set]) -> str:
    out: List[str] = []
    for S in sets_per_pos:
        if S == {'-'}: out.append('-'); continue
        if S <= {'I','-'}: out.append('I'); continue
        if S <= {'E','-'}: out.append('E'); continue
        if S <= {'X','-'}: out.append('X'); continue
        if S <= {'A','-'}: out.append('A'); continue
        if S <= {'D','-'}: out.append('D'); continue
        if S <= {'S','-'}: out.append('S'); continue
        if S <= {'H','-'}: out.append('H'); continue
        if 'S' in S: out.append('G'); continue
        if 'H' in S: out.append('C'); continue
        if ('A' in S and ('E' in S or 'I' in S)): out.append('M'); continue
        if ('D' in S and ('E' in S or 'I' in S)): out.append('L'); continue
        if {'E','I'} & S: out.append('X'); continue
        out.append('X')
    return ''.join(out)

def clean_hash_runs(seq: str, tchar: Dict[str,set]) -> str:
    """Replace '#' runs using neighboring context while enforcing valid transitions"""
    lst = list(seq); n = len(lst)
    def nearest_left(i: int):
        j = i-1
        while j>=0 and lst[j]=='#':
            j -= 1
        return lst[j] if j>=0 else None
    def nearest_right(j: int):
        k = j+1
        while k<n and lst[k]=='#':
            k += 1
        return lst[k] if k<n else None

    i = 0
    while i < n:
        if lst[i] != '#':
            i += 1; continue
        j = i
        while j < n and lst[j] == '#':
            j += 1
        L = nearest_left(i)
        R = nearest_right(j-1)
        fill = 'X'
        if L and R and (L in BACKGROUND) and (R in BACKGROUND) and L == R:
            fill = L
        elif L and L in BACKGROUND:
            fill = L
        elif R and R in BACKGROUND:
            fill = R
        if L and not is_valid_transition_char(tchar, L, fill):
            fill = 'X'
        if R and not is_valid_transition_char(tchar, fill, R):
            fill = 'X'
        for k in range(i, j):
            lst[k] = fill
        i = j
    return ''.join(lst)

def validate_path(annot: str, tchar: Dict[str,set]) -> bool:
    ok = True
    for i in range(len(annot)-1):
        a, b = annot[i], annot[i+1]
        if not is_valid_transition_char(tchar, a, b):
            logging.warning("Invalid transition %s->%s at pos %d", a, b, i)
            ok = False
    return ok

# ------------------- Main -------------------
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    target_fasta = Path(args.target_fasta)
    exonlist = Path(args.exonlist)
    s2t = Path(args.source2target)
    annot_out = Path(args.annotations_out)
    train_out = Path(args.training_out)
    states_path = Path(args.states)

    states, transitions = read_states(states_path)
    tchar = build_char_transitions(states, transitions)

    genes = read_genes(target_fasta)  # [(gene_id, dna)]
    exon_blocks = read_exonlist(exonlist)  # [[ENST, [<tx_start> <tx_end> <gene_start> <gene_end>], ...], ...]
    s2t_pairs = read_source2target(s2t)    # [(ENST, ENSG), ...]

    # Build transcript index aligned with source2target order
    tx_to_block: Dict[str, List[str]] = {blk[0]: blk for blk in exon_blocks}

    # Group transcripts by gene, s2t is sorted by gene
    groups: List[Tuple[str,List[str]]] = []  # [(ENSG, [ENST,...]), ...]
    current_gene, bucket = None, []
    for tx, g in s2t_pairs:
        if current_gene is None:
            current_gene = g
        if g != current_gene:
            groups.append((current_gene, bucket))
            current_gene, bucket = g, []
        bucket.append(tx)
    if current_gene is not None:
        groups.append((current_gene, bucket))

    # Build per-gene tracks: [DNA, ENST1_ann, ENST2_ann, ...]
    per_gene_tracks: Dict[str, List[str]] = {}
    gene_seq_map = dict(genes)

    missing_genes: List[str] = []  # collect genes not found in FASTA
    missing_txs:   List[str] = []  # collect transcripts missing in exonlist

    for g_id, txs in groups:
        dna = gene_seq_map.get(g_id)
        if dna is None:
            logging.warning("Gene %s in source2target not found in FASTA — skipped.", g_id)
            missing_genes.append(g_id)
            continue

        tracks = [dna]
        for tx in txs:
            blk = tx_to_block.get(tx)
            if blk is None:
                logging.warning("Transcript %s for gene %s not present in exonlist; skipped.", tx, g_id)
                missing_txs.append(tx)
                continue
            ann = annotate_transcript(len(dna), blk[1:])
            tracks.append(ann)

        per_gene_tracks[g_id] = tracks

    # write a small report of missing items
    try:
        Path("results").mkdir(parents=True, exist_ok=True)
        if missing_genes:
            with open("results/missing_genes.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(set(missing_genes))) + "\n")
            logging.info("Wrote list of missing genes to results/missing_genes.txt (%d entries).", len(set(missing_genes)))
        if missing_txs:
            with open("results/missing_transcripts.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(set(missing_txs))) + "\n")
            logging.info("Wrote list of missing transcripts to results/missing_transcripts.txt (%d entries).", len(set(missing_txs)))
    except Exception as e:
        logging.warning("Could not write missing items report: %s", e)


    # Build consensus per gene
    with annot_out.open("w", encoding="utf-8") as fout:
        for g_id, dna in genes:
            tracks = per_gene_tracks.get(g_id, [dna])
            # pad or skip if only DNA
            pos_sets = per_position_sets(tracks)
            gene_cons = consensus_from_sets(pos_sets)
            gene_cons = clean_hash_runs(gene_cons, tchar)
            if not validate_path(gene_cons, tchar):
                gene_cons = clean_hash_runs(gene_cons, tchar)  # second pass if needed

            # write: >GENE_ (DNA) ; >GENE (consensus) ; >ENST (each)
            fout.write(f">{g_id}_\n{dna}\n>{g_id}\n{gene_cons}\n")
            for i, txann in enumerate(tracks[1:], start=0):
                # track[1:] order follows groups; need the TXIDs again
                # Here we reconstruct names from s2t
                # iterate the s2t for g_id in order:
                for tx, g in s2t_pairs:
                    if g == g_id:
                        # write only once per transcript (first match)
                        fout.write(f">{tx}\n{txann}\n")
                        break

    # Build training pairs
    with train_out.open("w", encoding="utf-8") as tout, annot_out.open("r", encoding="utf-8") as fh:
        it = iter(fh.readlines())
        while True:
            try:
                header_adn = next(it).rstrip("\n")
                if not header_adn.startswith(">") or not header_adn.endswith("_"):
                    continue
                seq_adn = next(it).rstrip("\n")
                header_ann = next(it).rstrip("\n")
                seq_ann = next(it).rstrip("\n")
                tout.write(header_adn + "\n" + seq_adn + "\n")
                tout.write(header_ann + "\n" + seq_ann + "\n")
            except StopIteration:
                break

    logging.info("Done. Wrote %s and %s", annot_out, train_out)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)