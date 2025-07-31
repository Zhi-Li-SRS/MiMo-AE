# Future Work Roadmap

This document lists potential enhancements beyond the current prototype from **Data Pipeline**, **Model Improvement**, and **Evaluation**.

## 1. Data Pipeline  🔬
### 1.1 Advanced Pre-processing
- **Artifact detection & removal**  
  • Detect spikes, saturation, or sensor drop-outs with robust z-scores or self-supervised models and remove or inpaint the affected segments.
- **Baseline-wander correction**  
  • Apply an adaptive high-pass filter (cut-off ≈ 0.05 Hz) to BP and PPG to eliminate low-frequency drift while preserving pulse shape.
- **Cross-channel synchronisation**  
  • Realign channels based on detected R-peaks / respiratory peaks to minimise phase error.

### 1.2 Adaptive / Multi-rate Sampling
- **Channel-specific rates**  
  • Keep breath at 25–30 Hz, resample BP & PPG to 60–120 Hz, then up-sample or fuse at the network input.
- **Learnable down-sampling**  
  • Use a trainable Sinc-Conv or FIR layer as an end-to-end anti-alias filter.

### 1.3 Intelligent Windowing
- **Event-triggered windows**  
  • Anchor segments on heartbeat or breathing peaks instead of using a blind sliding window to reduce phase noise.
- **Variable-length windows**  
  • Allow 4- to 10-second inputs and unify them with positional encodings or adaptive pooling for greater robustness.

### 1.4 Monitoring & Versioning
- **Drift detection in production**  
  • Monitor distribution shift and automatically trigger μ/σ recomputation or model fine-tuning.
- **Version-controlled pipeline**  
  • Employ DVC or LakeFS to maintain full provenance from raw to training data.

---
## 2. Model & Training  ⚙️
- Multi-modal aligned autoencoder (cross-attention / transformer encoder).  
- Contrastive pre-training followed by autoencoder fine-tuning.  
- Knowledge distillation to a lightweight model for edge deployment.

## 3. Evaluation  📈
- Clinical relevance metrics: SBP/DBP reconstruction error, heart-rate variability, etc.  
- Self-supervised metrics such as VICReg to measure latent consistency.  
- End-to-end A/B testing in real-world scenarios.

---
> Prioritise the steps that most improve signal quality (sections 1.1 – 1.2) before extending the model itself.
