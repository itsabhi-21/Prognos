# Prognos: Remaining Useful Life Prediction for Turbofan Engines

Predicting how many cycles a jet engine can still run from its sensor history (NASA C-MAPSS), with an end-to-end pipeline: data verification, EDA, preprocessing, baselines, deep sequence models, hyperparameter tuning, multi-seed evaluation, error analysis and an interactive demo.

**Live demo:** <STREAMLIT_LINK>

## Why this matters
Unplanned engine failures are expensive and unsafe, while replacing parts too early wastes money. A Remaining Useful Life (RUL) estimate lets maintenance be scheduled at the right time.

## Data
NASA C-MAPSS Turbofan Engine Degradation Simulation (Saxena and Goebel, 2008), four subsets:

| Subset | Train / test engines | Operating conditions | Fault modes |
|---|---|---|---|
| FD001 | 100 / 100 | 1 | 1 |
| FD002 | 260 / 259 | 6 | 1 |
| FD003 | 100 / 100 | 1 | 2 |
| FD004 | 249 / 248 | 6 | 2 |

The data is not included in the repo. Download it from the NASA Prognostics Data Repository (or a Kaggle mirror) and place the files in `data/raw/`.

## Approach
1. **Verification and labels** (`01`): no missing values or duplicates, continuous cycles; RUL target created and capped at 125.
2. **EDA** (`02`): sensor degradation trends, correlations, K-Means confirms the 6 operating conditions of FD002/FD004; per-condition normalisation shown to be necessary.
3. **Preprocessing and baselines** (`03`): sensor selection per subset, per-condition normalisation (fitted on training data only), rolling features, engine-wise validation split, Ridge / Random Forest / XGBoost, 30-cycle sliding windows.
4. **Deep models** (`04`): LSTM, GRU, 1D-CNN.
5. **Tuning** (`05a`): Optuna (25 trials per subset) on validation engines only.
6. **Benchmark and analysis** (`05b`): 5 seeds, ensemble, error analysis, s9/s14 ablation.
7. **Demo** (`06`, `app/`): Streamlit app with RUL curves and model-spread band.

## Results
Evaluation: one prediction per test engine at its last recorded cycle against the true RUL (lower is better). GRU: mean ± std over 5 seeds.

| Subset | Best untuned baseline (RMSE) | GRU RMSE | GRU NASA score |
|---|---|---|---|
| FD001 | 20.15 (XGBoost) | 15.64 ± 0.27 | 590 ± 89 |
| FD002 | 28.17 (Random Forest) | 26.26 ± 0.20 | 6,587 ± 296 |
| FD003 | 21.41 (XGBoost) | 14.72 ± 0.76 | 513 ± 198 |
| FD004 | 29.64 (Random Forest) | 27.01 ± 0.68 | 5,812 ± 1,076 |

**Findings**
- The GRU lowers the RMSE by about 22%, 7%, 31% and 9% compared with the best baseline and lowers the NASA score on every subset.
- Accuracy is highest where it matters: RMSE of 2.5 to 6.4 cycles when fewer than 30 cycles remain. The main weakness is late warnings at medium RUL (60-100 cycles).
- The large RMSE on FD002/FD004 is mostly caused by the RUL cap: against the capped truth the RMSE is about 14-15.
- Hyperparameter tuning did not improve the test results beyond seed noise; the configuration was chosen from validation results before looking at the test set.
- The engine-specific sensors s9 and s14 are not needed (ablation).
- Compared with the literature the results are comparable to classic LSTM / CNN-LSTM models but behind recent attention-based models, especially on FD002/FD004. Published results use different protocols (window length, RUL cap, evaluation convention), so this is only a rough positioning.

## Run it
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```
The notebooks (`notebooks/`) were run on Google Colab (GPU for notebooks 04 to 06), in numerical order.

## Limitations
- Only the GRU family was tuned, and windows were limited to at most 30 cycles.
- Dropout is inactive for the single-layer models (FD003, FD004).
- Results come from the simulated C-MAPSS data, not from real engines.

## Reference
A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.