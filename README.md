# Selective perturbation of mirror and non-mirror neurons in an in silico model of the Action Observation Network

**Guglielmi et al.**

This repository contains the TensorFlow 2 implementation of the recurrent rate-based neural network model described in
*Selective perturbation of mirror and non-mirror neurons in an in silico model of the Action Observation Network* (Guglielmi et al., 2025).

The model was developed to reproduce and extend the neurophysiological findings of Ferroni et al. (2021, Current Biology 31, 2819–2830.e4) by training biologically constrained recurrent neural networks (RNNs) to match single-neuron firing patterns across three key nodes of the macaque Action Observation Network (AIP, F5, and F6).
This framework enables in silico causal perturbations of specific neuronal populations (mirror vs. non-mirror; excitatory vs. inhibitory) that are currently not experimentally accessible in vivo.

The experimental dataset used for model training and validation is publicly available in:
Tili et al. (2025), “Mirror neurons in monkey frontal and parietal areas”, Scientific Data, DOI: 10.1038/s41597-025-05299-9.

---

## 🔍 Overview

- Continuous-time rate-based RNN implemented in TensorFlow 2  
- Gradient clipping and early stopping  
- Separate inhibitory/excitatory structure loaded from `InhibitoryMask.mat` mask
- Requires experimental firing rates (`Firing_Rates.mat`) for supervision  
- Neuron identity file (`Mirror.mat`) containing mirror / non-mirror labels

---

## 🧩 Requirements

- Python ≥ 3.9  
- TensorFlow ≥ 2  
- NumPy, SciPy, Matplotlib (optional)

---

## 🚀 Usage example

Train the model:
```bash
python3 main_tf2.py   --firing_rates_mat ./Firing_Rates.mat   --inhibitory_mask_mat ./InhibitoryMask.mat   --mode train   --n_trials 1000   --learning_rate 1e-3   --loss_fn l2   --N1 86 --N2 106 --N3 163   --thermal 100
```

All outputs (checkpoints, logs, and configuration) are saved in `out_tf2/`.

---

## 📂 Repository structure

```
main_tf2.py        # Training pipeline
rnn_rate_tf2.py    # Core RNN definition and loss
Firing_Rates.mat   # Experimental firing rates
InhibitoryMask.mat # Binary inhibitory/excitatory mask
Mirror.mat         # Mirror / non-mirror neuron labels
out_tf2/           # Automatically created output directory
```

---

## ⚙️ Citation

If you use or adapt this code, please cite:

> Guglielmi et al.  
> *Selective perturbation of mirror and non-mirror neurons in an in silico model of the Action Observation Network*.
