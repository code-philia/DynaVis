# DynaVis

DynaVis is a lightweight and modular framework for **training-dynamics visualization**, designed to map high-dimensional model representations to a 2D space while preserving both **spatial** and **temporal** consistency.  
This version contains all core components needed for training, exporting embeddings, and running comprehensive evaluation (NPR, motion consistency, neighbor-change).

---

## 📁 Project Structure

```
DynaVis_anonymous/
│
├── scripts/
│   ├── data_utils.py               # Data loading, normalization, sampling utilities
│   ├── eval_io.py                  # Input I/O helpers for evaluation
│   ├── eval_motion.py              # Motion–semantic consistency metrics
│   ├── eval_neighbor_change.py     # Neighbor-change preservation metrics
│   ├── eval_npr.py                 # NPR (Neighbor Preserving Rate) evaluation
│   ├── export_utils.py             # Embedding export functions
│   ├── hparams.py                  # Hyper-parameter definitions
│   ├── losses_motion.py            # Loss functions for DynaVis training
│   ├── models.py                   # Encoder / Decoder model definitions
│   ├── train_motion.py             # Training loop for Stage-I / Stage-II
│   ├── utils_amp.py                # AMP utilities (mixed precision)
│
├── main.py                         # Entry point for training DynaVis
├── evaluation.py                   # Unified evaluation script (NPR + motion + neighbor-change)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔧 Environment Setup

We recommend creating a clean conda environment:

```bash
conda create -n dynavis python=3.9
conda activate dynavis
```

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training DynaVis

Run training with:

```bash
python main.py     --data_path /path/to/embeddings     --save_dir ./runs/exp1     --stage1_epochs 20     --stage2_epochs 20
```

All training logic is modularized inside:

```
scripts/
    train_motion.py
    losses_motion.py
    models.py
    hparams.py
```

---

## 📊 Evaluation Pipeline

Run full evaluation:

```bash
python evaluation.py     --hd_file exported/X_hd.npy     --ld_file exported/Y_ld.npy     --save_dir ./eval_results     --k 15     --t_start 1     --t_end 100
```

Metrics included:
- NPR (Neighbor Preserving Rate)
- Motion–semantic consistency
- Neighbor-change preservation

All results output as CSV + JSON.

---

## 📌 Notes

- This version removes dataset-specific code for anonymity.
- All components use modular, pluggable design for easy extension.
- Visualization rendering can be added on top of exported embeddings.

---

## 📄 License

This anonymous version is provided for **review / reproducibility** only.
