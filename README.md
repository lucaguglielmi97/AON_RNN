# Selective perturbation of mirror and non-mirror neurons in an in silico model of the Action Observation Network

**Guglielmi et al.**

This repository provides the TensorFlow 2 implementation of the recurrent rate-based neural network described in  
*Selective perturbation of mirror and non-mirror neurons in an in silico model of the Action Observation Network* (Guglielmi et al.).

The model reproduces the dynamics of the Action Observation Network for both execution and observation of a Go/No-Go reaching grasp using a continuous-time rate RNN with Dale's principle, task-specific input patterns, and neuron-wise adaptive time constants.

---

## 🔍 Overview

- Continuous-time rate-based RNN implemented in TensorFlow 2  
- Gradient clipping and early stopping  
- Separate inhibitory/excitatory structure loaded from `.mat` masks  
- Requires experimental firing rates (`Firing_Rates.mat`) for supervision  
- Requires neuron identity file (`Mirror.mat`) containing mirror / non-mirror labels

---

## 🧩 Requirements

- Python ≥ 3.9  
- TensorFlow ≥ 2.12  
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
