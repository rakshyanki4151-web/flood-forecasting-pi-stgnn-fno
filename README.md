# Hydrometeorological River Flow Forecasting & Flood Anomaly Detection

**Coventry University — STW7088CEM: Neural Networks & Systems**

---

## Project

A Hybrid Physics-Informed Spatio-Temporal Graph Neural Network and Fourier Neural Operator Framework for Daily Multi-Node River Flow Forecasting, Flood Anomaly Detection, and Explainable Early Warning.

---

## Repository Structure

```
├── code.ipynb                              # Complete project notebook (all 50 cells, full outputs)
├── app.py                                 # Interactive HydroAI Decision Support Dashboard (Figure 6.1)
├── requirements.txt                        # Runtime dependencies
├── final_selected_nodes.csv               # 345 sub-basin catchment IDs
├── optimized_figures_ultra/                # Ultra high-resolution publication figures (72 figures)
├── reproducibility_evidence/               # Cloud execution screenshots (RunPod A100-80GB)
│   ├── ss_phase1_env.png                  # Hardware & OS diagnostic
│   ├── ss_phase2_dag.png                  # Watershed graph & topological invariants
│   ├── ss_phase2_split.png                # Leakage-free chronological splits
│   ├── ss_phase3_loss.png                 # PhysicsInformedLoss compilation
│   ├── ss_phase3_params.png               # Parameter counts across 5 architectures
│   ├── ss_phase4_vae.png                  # VAE training & threshold calibration
│   └── runpod_interface_crop.png          # Cloud container session verification
└── outputs/                               # Complete outputs from cloud execution
    ├── figures/                           # Publication figures (.png, .pdf, .svg)
    ├── tables/                            # Evaluation tables (.csv and .xlsx, all 41 tables)
    ├── metrics/                           # Training histories, Optuna database, alerts
    └── reports/                           # experiment_config.json, reproducibility.json
```

---

## Hardware & Reproducibility

All models were trained on **RunPod** in a dedicated Linux container:

- OS: Linux 6.8.0-110-generic (x86_64)
- GPU: NVIDIA A100-SXM4-80GB (79.25 GB VRAM)
- CPU: 255 logical cores, 2003.84 GB Host RAM
- Software: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8
- Random seed: 42 (fixed across NumPy, PyTorch CPU, and PyTorch CUDA)

---

## Running the Code

```bash
git clone https://github.com/rakshyanki4151-web/flood-forecasting-pi-stgnn-fno.git
cd flood-forecasting-pi-stgnn-fno
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab
```

Open `code.ipynb` and run cells sequentially (Cell 1 to Cell 50).

### Interactive Decision Support Dashboard (Figure 6.1)

To launch the real-time HydroAI dashboard:
```bash
streamlit run app.py
```

---

## Results Summary

Test partition (2015–2017), 345 monitoring stations:

| Model | Parameters | RMSE | NSE | KGE |
| :--- | :---: | :---: | :---: | :---: |
| Naive Persistence | 0 | 0.942 | 0.412 | 0.435 |
| Baseline LSTM | 68,355 | 0.614 | 0.728 | 0.741 |
| Baseline GCN | 5,955 | 0.589 | 0.751 | 0.763 |
| Baseline STGNN | 56,003 | 0.521 | 0.812 | 0.825 |
| **Hybrid PI-STGNN-FNO** | **297,670** | **0.432** | **0.887** | **0.894** |

---

## License

MIT License. Dataset: LamaH-CE (Klingler et al., *Earth System Science Data*, 2021).
