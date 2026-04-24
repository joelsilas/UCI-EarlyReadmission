# UCI-EarlyReadmission
Aiming for Highest statistically honest, AUROC/AUPRC for the UCI Diabetes readmission dataset

---

## Benchmark Comparison (Statistically Honest, UCI Dataset Only)

> The majority of published papers on this dataset report AUROC > 0.85 or accuracy > 92 %. These are almost universally the result of data leakage — most commonly SMOTE applied before the train/test split, encounter-level (rather than patient-level) splitting, or retaining hospice/death discharges. Only papers that clearly avoid these pitfalls are listed below.
>
> `—` indicates the metric was not reported. † F1 and Accuracy for Liu 2024 are measured on SMOTE-balanced evaluation folds, not the original imbalanced distribution.

| Name / Paper | Best Model | AUROC | AUPRC | Accuracy | F1 (minority) |
|---|---|:---:|:---:|:---:|:---:|
| **Ours (5-seed mean ± std)** | **(see Results)** | **0.6824 ± 0.0067** | **0.2329 ± 0.0094** | **—** | **0.2942 ± 0.0086** |
| Emi-Johnson & Nkrumah, *Cureus* (2025) | XGBoost | 0.667 | — | — | — |
| Liu et al., *J Med Artif Intel* (2024) | XGBoost + GWO | 0.64 | — | 0.88† | 0.84† |
| Shang et al., *BMC Med Inform* (2021) | Random Forest | 0.640 | — | — | — |
| Bhuvan et al., arXiv:1602.04257 (2016) | AdaBoost | 0.63 | — | — | — |
| Strack et al., *BioMed Res. Int---