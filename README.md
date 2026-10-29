# A Controlled Comparative Study of Domain-Adaptive Pre-training and Distillation for Biomedical Sentence Classification

> **Status:** 📝 Accepted at the IEEE 18th International Scientific Conference on Informatics (Informatics 2026), 18–20 November 2026, Poprad, Slovakia. The paper has not yet been published. The full citation and DOI will be added after publication.

## Overview

This project compares three fine-tuning approaches for sequential sentence classification (SSC) in biomedical abstracts: direct fine-tuning, fine-tuning with domain-adaptive pre-training (DAPT), and fine-tuning with soft knowledge distillation. All approaches are built on a shared contrastive pre-training (SimCSE) and local-context concatenation foundation. They are evaluated with bootstrap confidence intervals and paired significance testing across three biomedical pre-trained language models (PubMedBERT, BioLinkBERT, BioBERT) on the PubMed 20k RCT benchmark. An ablation without the SimCSE stage is also included.

| Approach (script name) | Description |
|---|---|
| Approach 1 (Config 1) | SimCSE-pre-trained encoder, then fine-tuning with local context and class-weighted focal loss |
| Approach 2 (Config 2) | SimCSE, then DAPT (masked language modeling on the unlabeled sentences), then fine-tuning |
| Approach 3 (Config 3) | Soft distillation: the fine-tuned Approach 2 model is calibrated by temperature scaling and frozen as the teacher; the student is initialized as a copy of it |
| Ablation | Approach 1 without SimCSE (original PLM weights), with focal loss and with class-weighted cross-entropy |

The script also contains a fourth configuration (supervised contrastive learning, "Config 4") that is not part of the paper and is not reported.

## Requirements

- Python 3 with a GPU (CUDA) recommended; the script also runs on CPU, but very slowly
- `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`
- `spacy` and `scispacy` with the `en_core_sci_md` model (used for sentence splitting of the unlabeled abstracts). If it is not available, the script falls back to a simple regex splitter, which may change the unlabeled sentences slightly.

```bash
pip install torch transformers scikit-learn pandas numpy tqdm spacy scispacy
pip install <path-or-url-to>/en_core_sci_md-0.5.4.tar.gz
```

## Data

1. **Labeled data:** the PubMed 200k RCT corpus by Dernoncourt and Lee (https://github.com/Franck-Dernoncourt/pubmed-rct). Only the 20k version is used: `PubMed_20k_RCT/train.txt`, `dev.txt`, and `test.txt`. Abstract boundaries are taken from the original `###PMID` lines.
2. **Unlabeled data:** a CSV file with an `Abstract` column containing recent PubMed RCT abstracts (9,996 retrieved with a topic-restricted query; 5,000 are sampled with seed 42). None shares a PubMed identifier with the benchmark.

## Usage

The script was developed as a Google Colab notebook and exported to a Python file. Set the paths with environment variables (or edit the defaults at the top of the script):

```bash
export DATA_DIR=/path/to/PubMed_20k_RCT      # contains train.txt, dev.txt, test.txt
export UNLAB_CSV=/path/to/unlabeled_abstracts.csv
export SAVE_DIR=/path/to/output_folder        # checkpoints, predictions, and CSV results
python <script_name>.py
```

Notes:

- The exported file still contains Colab-specific lines (Google Drive mounting and `!pip install ...`). When running outside Colab, remove or comment out the `!pip install` line, since it is notebook syntax.
- **Choosing the backbone:** set `selected_model` near the top of the script to `"pubmedbert"`, `"biolinkbert"`, or `"biobert"`, and run once for each. The paper reports all three.
- **SimCSE ablation:** it is enabled by default (`RUN_SIMCSE_ABLATION=1`). Set the variable to `0` to skip it.
- **Resuming:** best checkpoints are saved in `SAVE_DIR` with the backbone name in the file name. If a checkpoint exists, that stage is skipped and the model is loaded. Delete the checkpoint to retrain it.
- **Randomness:** a single seed (42) is used throughout. Results can vary slightly across hardware and library versions.

## Implementation details

- All text, labeled and unlabeled, is lowercased.
- Local context (previous sentence, target sentence, next sentence joined with `[SEP]`) is used for the labeled data only. The unlabeled sentences (SimCSE, DAPT, distillation) are used without context.
- Hyperparameters: SimCSE (batch 128, 5 epochs, lr 3e-5, τ = 0.05); fine-tuning (batch 128, up to 20 epochs, lr 3e-5 or 5e-6, patience 5, focal loss γ = 2, class weights from the effective number of samples with β = 0.9999); DAPT (batch 64, 3 epochs, lr 5e-5, 15% masking); distillation (up to 6 epochs, lr 2e-6, patience 3, λ = 0.5). Only the teacher's logits are temperature-scaled.
- Evaluation: micro-F1 and macro-F1 with 90% bootstrap confidence intervals (B = 1,000), and paired bootstrap differences against Approach 1 on the same resampled indices. Test sentences are the resampling unit.

## Outputs

For each backbone `<plm>`, the script writes to `SAVE_DIR`:

| File | Content |
|---|---|
| `conference_ablation_results_<plm>.csv` | Micro-F1, macro-F1, 90% CIs, and per-class F1 for each configuration |
| `conference_pairwise_significance_<plm>.csv` | Paired bootstrap differences in micro-F1 (unadjusted 90% CIs) |
| `per_class_significance_<plm>.csv` | Paired bootstrap per-class F1 differences for Approaches 2 and 3 vs. Approach 1 |
| `simcse_ablation_<plm>.csv` | Paired differences for the no-SimCSE ablation, micro-F1 and macro-F1 |
| `preds_<plm>.npz` | Test-set predictions of every model |
| `*.pt` | Best checkpoints |

The script itself reports unadjusted bootstrap intervals. The saved results of the runs reported in the paper are provided in `results/` (see below).

## Results

The saved results of the runs reported in the paper are in `results/`, so they can be inspected without retraining.

| File | Content |
|---|---|
| `preds_<plm>.npz` | Test-set predictions of every model (30,135 sentences) |
| `conference_ablation_results_<plm>.csv` | Micro-F1, macro-F1, 90% bootstrap CIs, and per-class F1 |
| `conference_pairwise_significance_<plm>.csv` | Paired bootstrap difference in micro-F1 between configurations (unadjusted 90% CIs) |
| `per_class_significance_<plm>.csv` | Paired bootstrap per-class F1 differences for Approaches 2 and 3 vs. Approach 1 |
| `simcse_ablation_<plm>.csv` | Paired differences for the no-SimCSE ablation (micro-F1 and macro-F1) |

`<plm>` is `pubmedbert`, `biolinkbert`, or `biobert`.

**Reading the prediction files.** Each `.npz` file holds the true labels (`y_true`) and the predicted labels of each model as integer class ids:

```python
import numpy as np
from sklearn.metrics import f1_score

z = np.load("results/preds_pubmedbert.npz")
print(z.files)  # y_true, config1, config2, config3, config4, ablationA, ablationB
print(f1_score(z["y_true"], z["config1"], average="micro"))
```

| Key | Model |
|---|---|
| `config1` | Approach 1 (direct fine-tuning) |
| `config2` | Approach 2 (+ DAPT) |
| `config3` | Approach 3 (+ soft distillation) |
| `ablationA` | Approach 1 without SimCSE (focal loss) |
| `ablationB` | Approach 1 without SimCSE (class-weighted cross-entropy) |
| `config4` | Supervised contrastive variant, not reported in the paper |

Class ids follow the alphabetical order of the labels: BACKGROUND, CONCLUSIONS, METHODS, OBJECTIVE, RESULTS.

**Note on signs.** In `simcse_ablation_<plm>.csv` the reported difference is Approach 1 minus the no-SimCSE variant. Table III of the paper reports the variant minus Approach 1, so the signs are opposite.

The paired bootstrap settings are those in the paper (B = 1,000, seed 42, test sentences as the resampling unit). The Holm-adjusted p-values and McNemar's test in Table III were computed from these prediction files.

## Authors

- Zhyar Rzgar K. Rostam
- Gábor Kertész

Óbuda University, Budapest, Hungary

## Citation

The paper has been accepted at Informatics 2026 and has not yet been presented or published. The citation and DOI will be added after publication.