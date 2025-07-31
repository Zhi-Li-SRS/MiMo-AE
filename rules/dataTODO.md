# Data Pre-processing TODO

This checklist covers all pre-processing steps required before training the multimodal auto-encoder described in *task.pdf*.  The raw material is the **10 cleaned *.csv*** files found in `cleaned_data/`, each representing one subject and already trimmed to the 3 signals of interest (`bp`, `breath_upper`, `ppg_fing`).

---
## 0.  Subject split (fixed up-front)
| Role  | Subjects (file stem) |
|-------|----------------------|
| **train (7)** | STLE_0125, STLE_0094, STL_0053, STLE_0099, STLE_0250, STLW_0341, STLE_0140 |
| **eval  (1)** | STLE_0223 |
| **test  (2)** | SF_0034, STLE_0372 |

*Rationale*: Keep the test set completely unseen; pick one subject for early-stopping / hyper-param tuning; the remaining seven for training.
---
## 1.  Exploratory sanity check  ✅
- [ ] Verify sampling rate per file from the `time` column (expected ≈ 2 kHz from original BioPac export).
- [ ] Plot short snippets of each signal to visually confirm integrity & alignment.

---
## 2.  Down-sample to 30 Hz  ⏬
- [ ] Choose target rate: **30 Hz** (240 samples for an 8 s window).
- [ ] Apply anti-aliasing low-pass filter (e.g., `scipy.signal.decimate` or polyphase `resample_poly`).
- [ ] Down-sample **all three channels simultaneously** to keep alignment.
- [ ] Store the result as a NumPy array of shape `(N_samples_ds, 3)` for each subject.

---
## 3.  Window segmentation  📏
- [ ] Segment each subject’s down-sampled array into **8-second windows** (240 frames).
- [ ] Decide on stride:  
  	• *Option A*: non-overlapping (stride = 240) — simplest.  
  	• *Option B*: 50 % overlap (stride = 120) — more training examples.
- [ ] Discard final partial window if shorter than 8 s.
- [ ] Result: for every subject prepare an array `(n_windows, 3, 240)`.

---
## 4.  Normalisation  ⚖️
- [ ] Compute **mean** and **std** of each channel **using only the training subjects**.
- [ ] Save stats to `stats.json` for reproducibility.
- [ ] Apply (x − μ)/σ transformation to *all* splits with the same μ, σ values.

---
## 5.  Data packaging  📦
- [ ] For each split create a single compressed file, e.g.  
  	`processed/train_windows.npz` → `windows` array, shape `(M_train, 3, 240)`
- [ ] Optionally store per-subject indices if needed for subject-level evaluation.

---
## 6.  PyTorch‐friendly dataset  🔌
- [ ] Implement `Dataset` class that 
  ```python
  item ⇒ (torch.FloatTensor shape [3, 240])
  ```
- [ ] Provide one class per split or pass split name to `__init__`.
- [ ] Wrap with `DataLoader` (shuffle=True for train).

---
## 7.  Verification & visualisation  🔍
- [ ] Plot random normalised window to confirm range ~N(0,1).
- [ ] Check no NaNs / Infs after processing.
- [ ] Ensure train/eval/test windows counts look reasonable.

---
## 8.  Automation  🛠️
- [ ] Write a single script `prepare_data.py` that runs **all steps 1-7**.
- [ ] Accept `--raw_dir`, `--out_dir`, `--target_rate`, `--window_len`, `--stride` CLI args.

---
## 9.  Documentation  📝
- [ ] Update project README with exact commands to reproduce the processed dataset.

> Once every box above is ticked, the data pipeline is ready for model training.
