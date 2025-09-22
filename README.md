# Time-Frequency-Fusion: Parallel Time-Frequency Domain Forecasting Architecture

Time-Frequency-Fusion is a neural network architecture designed for multivariate time series forecasting. It leverages **multi-scale temporal decomposition**, **frequency domain enhancements**, and **cross-domain attention-based fusion** to effectively model both short-term and long-term dependencies.

---

## 🧠 Key Features

- **Reversible Instance Normalization (RevIN):** Handles distribution shifts in input sequences.
- **Time Domain Branch:**
  - Decomposes sequences into trend and seasonal components using multi-scale moving averages.
  - Applies linear layers (MLPs) independently or jointly on features.
- **Frequency Domain Branch:**
  - Applies FFT and learns to enhance key frequency bands.
  - Computes periodicity scores to inform the fusion step.
- **Cross-Domain Fusion:**
  - Uses multi-head attention between time and frequency branches.
  - Learns to adaptively fuse both domains using periodicity-based gates.
- **Modular & Configurable:** Easily switch between time-only and time-frequency parallel branches.
