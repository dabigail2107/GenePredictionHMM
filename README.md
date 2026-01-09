#  Eukaryotic Gene Prediction using H(S)MMs with Priors
This repository implements a semi-Markov gene prediction pipeline based on probabilistic models combining intrinsic sequence features and structural priors. It includes utilities for data preparation, training, prediction, and evaluation of Hidden Semi-Markov Models (H(S)MM) for gene structure inference.

---

## 📁 Project Structure

```text
project_root/
├── model.json # Trained model (output of training)
├── data/ # Input data (user-provided; examples below from ENSEMBL 98)
│ ├── human_target.fasta # Target genomic DNA per gene (FASTA)
│ ├── human_initialsourceexonlist.txt # Transcript exon coordinates (per transcript)
│ └── initialsource2target.fasta # Transcript→Gene mapping
├── source/
│ ├── format_data.py # Build paired DNA/annotation FASTA from source tables
│ ├── split_data.py # Split FASTA into train/test sets
│ ├── train_with_priors.py # Train semi-Markov model with priors + observed counts
│ ├── predict.py # Single-sequence gene prediction
│ ├── batch_predict.py # Batch prediction on multi-FASTA
│ ├── eval_states.py # Evaluate predicted vs. true annotations
│ └── priors.json # Prior model configuration (motifs, transitions, lengths)
├── results/
│ ├── annotations_out.fasta # (from format_data) multi-track per gene
│ ├── trainingData_out.fasta # (from format_data) paired DNA/annotation
│ ├── train_out.fasta # Training set (DNA + annotation)
│ ├── test_out.fasta # Test set (DNA only)
│ ├── test_truth.fasta # Ground truth for test
│ └── test_predicted.fasta # Predicted states (batch)
└── README.md
```


---

## ⚙️ Installation

- Python 3.8+
- `pip install numpy`

Optional (recommended): `git lfs install` if you plan to store large FASTA files.

---

## 🧩 End-to-End Workflow (single-line commands)

### 0️⃣ Prepare input FASTA (DNA + annotation)

`python source/format_data.py data/human_target.fasta data/human_initialsourceexonlist.txt data/initialsource2target.fasta results/annotations_out.fasta results/trainingData_out.fasta --states source/states.json`

Outputs:

results/trainingData_out.fasta → paired DNA/annotation
results/annotations_out.fasta → per-gene tracks
Missing or incomplete genes are skipped with warnings (the script does not abort).

---

### 1️⃣ Split into training and test sets


`python source/split_data.py results/trainingData_out.fasta --train-out results/train_out.fasta --test-out results/test_out.fasta --test-truth results/test_truth.fasta --ratio 0.8 --seed 42`

### 2️⃣ Train H(S)MM with priors + observed counts

`python source/train_with_priors.py results/train_out.fasta source/priors.json model.json --order 4 --blend-transitions 0.0 --blend-start 0.0 --gamma-scale 1.0 --min-bg-len 30`

Output:

model.json (at the project root)

### 3️⃣ Predict on a single FASTA sequence

`python source/predict.py model.json results/test_out.fasta --start-from-first-atg --stop-mode last --site-penalty 0.6 --beam 6.0 --letters results/pred_letters.txt --gff3 results/prediction.gff3 --bed results/prediction.bed`

Outputs:

pred_letters.txt → state letters
prediction.gff3, prediction.bed → structural annotations

### 4️⃣ Batch prediction (multi-FASTA)

`python source/batch_predict.py source/predict.py model.json results/test_out.fasta results/test_predicted.fasta --start-from-first-atg --stop-mode last --min-bg-len 30 --beam 20 --site-penalty 0.8`

Each sequence produces:

>ID|original → DNA
>ID|predicted_states → predicted letters over {S,G,E,I,X,A,M,D,L,H,C}

### 5️⃣ Evaluate predictions vs. ground truth

`python source/eval_states.py results/test_truth.fasta results/test_predicted.fasta --outdir results/eval`

Outputs:

results/eval/summary.json
results/eval/per_class_metrics.csv
results/eval/confusion_matrix.csv

Example console output:

[GLOBAL] n_positions=125340  accuracy=0.8793

[GLOBAL] exon_vs_nonexon: precision=0.8531  recall=0.8672  F1=0.8601  IoU=0.7570  accuracy=0.9240

🧠 Model Overview

```text
| Component           | Description                                                                                                    |
| States              | 11 biological states: cstart, cstartA, exon, exintron, accsite, accAlt, donsite, donAlt, intron, cstop, cstopA |
| Emissions           | Background (order-4 Markov) for exon/intron/exintron; motif priors for splice, start, and stop codons          |
| Lengths             | Geometric duration models per background state                                                                 |
| Transitions         | Learned from training annotations and smoothed with priors                                                     |
| Start probabilities | Can be fixed or blended via priors (default: high probability for cstart)                                      |
```
