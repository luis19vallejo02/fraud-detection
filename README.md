# Credit card fraud detection

Explore how fraud appears in a highly imbalanced set of card transactions. The project currently covers environment setup, data validation and exploratory data analysis (EDA).

## Step 0 · Set up the environment

The workflow was verified with Python 3.12 on macOS. With Git and Python installed, run the following:

```bash
git clone https://github.com/luis19vallejo02/fraud-detection.git
cd fraud-detection
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -c "import sklearn, xgboost, shap; print('ok')"
```

The final command should print `ok`. The [requirements file](requirements.txt) includes JupyterLab and the analysis dependencies.

## Step 1 · Download and inspect the data

Download [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and extract `creditcard.csv` into `data/raw/`. Create the directory from the repository root with `mkdir -p data/raw`. Git ignores `data/raw/` and `data/processed/`.

Worldline and the Machine Learning Group at Université Libre de Bruxelles (ULB) collected the dataset. It spans roughly two days and contains 31 columns. `Class` is the label (0 for legitimate, 1 for fraud), `Time` is the number of seconds since the first transaction, and `Amount` is the transaction amount. `V1` through `V28` are anonymized PCA components whose original business variables are unknown.

| Data | Transactions | Fraud cases | Fraud rate |
| --- | ---: | ---: | ---: |
| Original CSV | 284,807 | 492 | 0.1727% |
| After deduplication | 283,726 | 473 | 0.1667% |

The initial inspection found zero missing values and 1,081 exact duplicate rows. The notebook keeps the first occurrence of each duplicate in memory and preserves the original CSV. Removing duplicates before the future temporal split prevents identical rows from appearing in multiple sets.

Kaggle lists the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/) for the database and the [Database Contents License (DbCL)](https://opendatacommons.org/licenses/dbcl/1-0/) for its contents. The repository code is under the [MIT license](LICENSE).

## Step 2 · Reproduce the exploratory analysis

From the repository root with `.venv` active, start JupyterLab with `python -m jupyterlab`. Open [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) and run all six code cells in order. The first two inspect and deduplicate the data. The next four generate the figures in `reports/figures/`. The final Markdown cell summarizes the findings.

![Class balance after removing exact duplicates](reports/figures/01_class_balance.png)

| Figure | Finding |
| --- | --- |
| [Class balance](reports/figures/01_class_balance.png) | Predicting every transaction as legitimate on the original data yields 99.8273% accuracy and detects zero frauds. This imbalance motivates PR-AUC as the primary evaluation metric. |
| [Transaction amount by class](reports/figures/02_amount_by_class.png) | The fraud median was 9.82 versus 22.00 for legitimate transactions. Fraud had a higher mean (123.87 versus 88.41). The chart uses `log(1 + Amount)` to show the skewed distributions. |
| [Fraud rate by relative hour](reports/figures/03_fraud_by_hour.png) | Relative hour 2 had the highest observed rate, 1.45% (48 of 3,308 transactions). Hours are measured from the first transaction and grouped modulo 24. The roughly two-day window limits conclusions about weekly seasonality. |
| [Features ranked by absolute correlation](reports/figures/04_top_features.png) | `V17`, `V14`, `V12` and `V10` had the largest absolute Pearson correlations with the fraud label. The figure shows the six highest-ranked features. |
