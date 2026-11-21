# TRINETRA: Spatio-Temporal Graph-Sequence Learning for Next POI Prediction

This repository provides the TRINETRA implementation and experiment pipeline
associated with the study. It includes the custom preprocessing/indexing logic,
dual-graph construction, GATv2 graph encoder, GPT-2 sequence predictor,
multi-objective training loop, and ranking evaluation metrics.

## Code and data availability

The Foursquare NYC and Tokyo check-in datasets used in this study are publicly
available through [Dingqi Yang's Foursquare dataset repository](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
(accessed 12 July 2026). The dataset is not redistributed in this repository.

The sequence modeling backbone uses the open-source
[GPT-2 implementation from Hugging Face Transformers](https://github.com/huggingface/transformers).
The custom data preprocessing pipelines, indexing modules, core algorithm, and
evaluation scripts developed for this study are provided in this repository.

## Repository contents

The original `TRINETRA_revised.ipynb` notebook is included **unchanged** at `notebooks/TRINETRA_revised.ipynb`. It is retained as a notebook; it has not been converted, reformatted, or edited as part of this packaging. The standalone `src/trinetra.py` is included separately from the source code supplied for the reviewer-oriented release. These are separate entry points; check their configuration cells/sections before comparing runs.

```text
TRINETRA-Next-POI/
├── notebooks/
│   └── TRINETRA_revised.ipynb    # Original notebook, unchanged
├── src/
│   └── trinetra.py                 # Model, preprocessing, graphs, training, evaluation
├── scripts/
│   └── prepare_foursquare.py       # Raw TSMC2014 TSV -> model-ready CSV
├── data/
│   └── README.md                   # Dataset download and placement instructions
├── results/                        # Generated result CSVs (not tracked)
├── checkpoints/                    # Generated model checkpoints (not tracked)
├── docs/
│   └── REPRODUCIBILITY.md           # Experimental protocol and implementation notes
├── requirements.txt
├── .gitignore
├── CITATION.cff
└── README.md
```

## 1. Dataset download

Use the original TSMC2014 Foursquare NYC/TKY dataset from:

- **Original source:** [Dingqi Yang — Foursquare Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
- **Alternative mirror:** [Foursquare NYC and Tokyo Check-in Dataset (Kaggle)](https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset)

Use the TSMC2014 NYC and Tokyo files, commonly named:

- `dataset_TSMC2014_NYC.txt`
- `dataset_TSMC2014_TKY.txt`

Check the original source's terms and cite the dataset paper. The Kaggle copy is
a third-party mirror; verify that it corresponds to the same TSMC2014 release
before using it.

The original TSMC2014 record format has eight tab-separated fields:
`userId`, `venueId`, `venueCategoryId`, `venueCategory`, `latitude`,
`longitude`, `timezoneOffset`, and `utcTimestamp`.

## 2. Environment setup

Recommended Python: 3.10 or 3.11. Install PyTorch for the local CPU/CUDA
environment first, then install a compatible PyTorch Geometric version.

```bash
git clone <REPOSITORY_URL>
cd TRINETRA-Next-POI

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Install PyTorch using the official selector for your platform and CUDA runtime:
https://pytorch.org/get-started/locally/

Then install PyTorch Geometric following its compatibility instructions:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Install remaining packages:

```bash
pip install -r requirements.txt
```

Record exact package, CUDA, GPU, and driver versions when reporting results.
PyTorch/PyG binary compatibility depends on the installed environment.

## 3. Prepare NYC and TKY files

Place the downloaded raw files anywhere accessible, then convert each file to
the CSV schema expected by the model:

```bash
python scripts/prepare_foursquare.py \
  --input /path/to/dataset_TSMC2014_NYC.txt \
  --output data/foursquare_nyc.csv

python scripts/prepare_foursquare.py \
  --input /path/to/dataset_TSMC2014_TKY.txt \
  --output data/foursquare_tky.csv
```

The converter retains the fields used by the supplied training implementation:

```text
userId, venueId, venueCategory, latitude, longitude, utcTimestamp
```

It discards the original category ID and timezone-offset columns because the
provided model derives category indices from the category-name field and parses
the UTC timestamp directly.

## 4. Run experiments

Run NYC:

```bash
CSV_PATH=data/foursquare_nyc.csv python src/trinetra.py
```

Run Tokyo:

```bash
CSV_PATH=data/foursquare_tky.csv python src/trinetra.py
```

The `CSV_PATH` environment variable selects the input dataset. The script uses
CUDA when available. If more than one CUDA device is detected, it wraps the
joint model in `torch.nn.DataParallel` and sets batch size to 1024; otherwise
the batch size is 128. Reduce the batch size if GPU memory is insufficient.

## 5. Configuration

The experiment parameters are defined near the beginning of `src/trinetra.py`.

| Parameter | Supplied value |
|---|---:|
| Minimum user check-ins | 10 |
| Minimum POI visits | 10 |
| Maximum sequence length | 100 |
| Joint training epochs | 100 |
| Early-stopping patience | 5 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.3 |
| Contrastive loss weight (`LAMBDA_CL`) | 0.1 |
| Coordinate loss weight (`LAMBDA_LOC`) | 1.0 |
| Contrastive temperature | 0.07 |
| Number of spatial clusters | 50 |
| Spatial KNN neighbours | 10 |
| Embedding dimension | 128 |
| Evaluation cut-offs | 1, 5, 10, 20 |
| Random seeds | As per wish |
| Graph pre-training epochs | 400 |

Do not change these values when attempting to reproduce the supplied configuration.
If you run a modified configuration, report it separately.

## 6. Implemented pipeline

1. Load the selected city CSV and normalize the timestamp field.
2. Remove invalid timestamps.
3. Filter venues with fewer than 10 visits and users with fewer than 10 check-ins.
4. Encode users, venues, and venue categories; reserve venue index 0 for padding.
5. Sort each user's check-ins chronologically.
6. Generate per-user leave-last-two-out targets: the final check-in is the test
   target, the second-to-last is the validation target, and earlier eligible
   positions form training samples. Each sample uses up to the preceding 100
   interactions as context.
7. Build a transition graph using transitions found in training sample histories.
8. Build a spatial KNN graph from venue coordinates.
9. Cluster venue coordinates using KMeans and construct venue category/coordinate
   node features.
10. Pre-train the dual GATv2 encoder with a link-prediction objective.
11. Transfer graph embeddings to the venue embedding table.
12. Jointly train the graph encoder and GPT-2-based sequence predictor using
    next-venue classification, contrastive alignment, and coordinate regression.
13. Report Acc@K, MRR@K and mean Acc@K (`mAcc`) for K = 1, 5, 10, 20.

More implementation details and important evaluation caveats are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 7. Generated outputs

The training script writes:

- `best_gpt2_model.pth` — checkpoint saved when validation mAcc improves.
- `five_seed_per_run_results.csv` — final test metrics recorded for each seed.
- `five_seed_summary.csv` — mean and population standard deviation across seeds.

Move or copy generated outputs into `results/` and `checkpoints/` if you want
to archive them locally. These artifacts and raw datasets are excluded from Git.

## 8. Citation

**Dataset paper**

Dingqi Yang, Daqing Zhang, Vincent W. Zheng, and Zhiyong Yu. “Modeling User
Activity Preference by Leveraging User Spatial Temporal Characteristics in
LBSNs.” *IEEE Transactions on Systems, Man, and Cybernetics: Systems*,
45(1), 129–142, 2015.

**Software**

Please cite the associated TRINETRA paper using its final published bibliographic
details. The repository does not invent or assume a DOI, venue, or author list.

## 9. License

No software license has been assigned in this repository. Add a license only
after confirming the authors' rights to license the implementation. Dataset
terms are independent of the software license.
