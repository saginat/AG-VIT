# AG-ViT: Atlas-Guided Vision Transformer for Predicting Cognitive Decline

A 4D Vision Transformer for learning from resting-state fMRI (rs-fMRI) under limited data. AG-ViT is pretrained with an **atlas-guided** objective — an encoder–decoder reconstructs a brain atlas representation (regional time series) from a 4D scan — and the resulting embeddings are used to predict the onset of cognitive decline in patients with Alzheimer's disease (AD).

> Code accompanying our paper (under review). More details and trained models will be added upon acceptance.

## Method

AG-ViT is trained in two stages:

1. **Self-supervised pretraining.** An encoder maps a 4D fMRI scan to a compact bottleneck; a lightweight decoder reconstructs the Schaefer-2018 atlas representation (200 regions × time).
2. **Supervised head training.** The encoder is frozen and a small MLP head is trained on the bottleneck features to predict whether a patient's Clinical Dementia Rating (CDR) will increase within 1, 2, or 3 years.

An optional **Test-Time Adaptation** step refines the bottleneck per subject at inference.

## Repository Structure

```
AG-VIT/
├── configs/      # Configuration files
├── data/         # Data loading and atlas extraction
├── models/       # AG-ViT model definitions
├── training/     # Training and adaptation loops
├── evaluation/   # Metrics and analysis
├── scripts/      # Entry-point scripts
└── utils/        # Shared helpers
```


```

## Data

Experiments use the [ADNI](https://adni.loni.usc.edu/) rs-fMRI repository. ADNI data is access-controlled and not distributed here; request access from ADNI directly.

## License

Released under the [MIT License](LICENSE). © 2026 Sagi Nathan.
