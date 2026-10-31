# doggo-dataset

A fully offline, licence-clean image dataset for binary classification of
**dog vs non-dog** animal photographs. It is designed as a *fixed evaluation
dataset* for comparing a Vision Model baseline against a Vision Model + Text RAG
workflow on the same images.

## Counts

| Label | Species | Target | Actual |
|---|---|---:|---:|
| dog | dog | 100 | 100 |
| non_dog | wolf | 10 | 10 |
| non_dog | fox | 10 | 10 |
| non_dog | coyote | 10 | 10 |
| non_dog | dingo | 10 | 10 |
| non_dog | cat | 10 | 10 |
| non_dog | koala | 10 | 10 |
| non_dog | red_panda | 10 | 10 |
| non_dog | raccoon | 10 | 10 |
| non_dog | bear | 10 | 10 |
| non_dog | hyena | 10 | 10 |
| | **total** | **200** | **200** |

Validation: **PASS** — see `metadata/validation_report.md`.

## Label definition (important)

This dataset uses a practical binary definition, not biological taxonomy:

- `dog` — domestic dogs only (`images/dog/`).
- `non_dog` — every other selected species (`images/non_dog/<species>/`).

Wolf, fox, coyote and **dingo are explicitly labelled `non_dog`** for this
project's classification task, even though the dingo is closely related to the
domestic dog. Do not silently relabel them from taxonomy.

## Structure

```text
doggo-dataset/
├── images/
│   ├── dog/dog_001.jpg … dog_100.jpg
│   └── non_dog/
│       ├── wolf/  fox/  coyote/  dingo/  cat/
│       ├── koala/  red_panda/  raccoon/  bear/  hyena/
├── metadata/
│   ├── manifest.csv            # per-image source, licence, size
│   ├── species_summary.json    # counts + label definition
│   ├── validation_report.json  # automated validation checks
│   ├── validation_report.md
│   └── failed_or_replaced.csv  # rejected/replaced candidates
├── LICENSES/ATTRIBUTION.md
├── README.md
└── doggo-dataset.zip
```

## Sources and licences

All photographs come from [Wikimedia Commons](https://commons.wikimedia.org)
and are reused under **CC0, Public Domain, CC BY or CC BY-SA** licences only.
Images under NonCommercial or NoDerivatives terms, images with unclear
ownership, artwork, illustrations, maps, diagrams, screenshots and suspected AI
renders were rejected during collection. Full per-image attribution (author,
source page, licence, licence URL) is in `manifest.csv` and
`LICENSES/ATTRIBUTION.md`.

## Image selection

- real JPEG photographs, longest side ≥ 1280px at download time,
  shortest side ≥ 480px
- the target species is visible and identified by the Commons category it was
  taken from
- exact duplicates (SHA-256) and near-duplicates (dual 64-bit perceptual
  dHash on LANCZOS and BICUBIC downscales, distance ≤ 10)
  are rejected and replaced by the next valid candidate
- dog images are sampled across hundreds of breed categories for breed, size,
  coat, pose and environment diversity; each species is sampled across several
  Commons categories to vary camera angle, pose, habitat and lighting

## Intended use

Fixed evaluation set only. Recommended experiments:

1. **Baseline:** image → vision model → `dog` / `non_dog`
2. **RAG-assisted:** image → vision model → text RAG → vision model → final label

Both experiments must use the same images and the same vision model. Never
modify image contents or labels to improve model results.

## Rebuilding and re-validating

```bash
uv run python scripts/build_doggo_dataset.py                 # full rebuild (network)
uv run python scripts/build_doggo_dataset.py --validate-only # offline re-validation
```
