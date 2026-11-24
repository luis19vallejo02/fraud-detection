# Credit card fraud detection

Explore fraud in a highly imbalanced set of card transactions. The workflow covers data validation, exploratory data analysis (EDA), chronological preprocessing, model selection against a balanced logistic regression baseline, evaluation at a threshold chosen by business cost, SHAP explanations, and reproducible training from the command line.

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

The initial inspection found zero missing values and 1,081 exact duplicate rows. The notebook keeps the first occurrence of each duplicate in memory and preserves the original CSV. Step 3 applies the same deduplication before the temporal split so identical rows cannot appear in multiple sets.

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

## Step 3 · Split chronologically and preprocess

With the environment from Step 0 and the CSV from Step 1, open [notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb) in JupyterLab and run the cells in order. [src/data.py](src/data.py) loads the same 283,726 deduplicated rows used in Step 2 and sorts them by `Time`. The split uses the earliest approximately 70% of rows for training, the next 15% for tuning and threshold selection, and the last 15% for final evaluation.

The nominal cuts at 70% and 85% fall inside groups of transactions with the same `Time`. Each cut moves to the start of its group, so every transaction at a boundary second goes into the later set. The notebook verifies strict time separation and at least one fraud case in each set.

| Set | Transactions | Fraud cases | Fraud rate | `Time` range (seconds) |
| --- | ---: | ---: | ---: | ---: |
| Train | 198,607 | 366 | 0.1843% | 0–132,905 |
| Validation | 42,558 | 55 | 0.1292% | 132,906–151,319 |
| Test | 42,561 | 52 | 0.1222% | 151,320–172,792 |

[src/features.py](src/features.py) removes `Time` from model inputs, keeps all 28 PCA components, and applies `log1p` followed by `RobustScaler` to `Amount`. This reuses the amount transformation explored in Step 2. The scaler learns the median and interquartile range from training alone and uses those same values for validation and test. `Class` is kept separately as the label. The notebook verifies these transformations before fitting the baseline.

With 52 fraud cases in test, detecting one additional case changes test recall by about 1.9 percentage points. The final recall estimate is therefore sensitive to a small number of cases.

## Step 4 · Fit and evaluate the baseline

Run the remaining cells of [notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb). The notebook fits logistic regression with `class_weight="balanced"`, `max_iter=2000`, and `random_state=42` on the preprocessed training set. It passes the fraud scores on validation and test to the reusable functions in [src/evaluate.py](src/evaluate.py).

| Set | PR-AUC (average precision) | ROC-AUC | Recall at ≥80% precision | Recall at ≥50% precision | PR-AUC with constant scores |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.812824 | 0.980591 | 74.55% | 85.45% | 0.001292 |
| Test | 0.689759 | 0.976271 | 69.23% | 75.00% | 0.001222 |

PR-AUC is the primary ranking metric. The reference with constant scores equals each set's fraud prevalence. On test, the majority-class classifier achieves 99.8778% accuracy and detects zero of 52 frauds.

Recall at fixed precision is computed over the score cutoffs available in each set. For later model comparisons, use the same temporal splits and compare validation PR-AUC. Choose an operating threshold on validation and report its performance on test with that threshold held fixed.

## Step 5 · Compare and select fraud models

Run all cells in [notebooks/03_models.ipynb](notebooks/03_models.ipynb) with the project's `.venv` active and the CSV in `data/raw/`. The notebook reuses the Step 3 chronological split and training-fitted preprocessor. It compares all candidates on the same 42,558 validation transactions, including 55 frauds. The selected model's held-out evaluation and operating threshold are part of Step 6.

| Model | Validation PR-AUC (average precision) | ROC-AUC | Recall at ≥80% precision |
| --- | ---: | ---: | ---: |
| XGBoost, tuned | **0.864248** | 0.988384 | 81.82% |
| XGBoost, default experiment | 0.853195 | 0.987224 | 78.18% |
| Balanced logistic regression | 0.812824 | 0.980591 | 74.55% |
| LightGBM | 0.386226 | 0.907797 | 0.00% |
| Isolation Forest | 0.033687 | 0.946454 | 0.00% |

Tuned XGBoost improved validation PR-AUC by 0.051424 and recovered 45 of 55 validation frauds at ≥80% precision. The best of 20 trials used depth 6, learning rate 0.01, minimum child weight 5, row sampling 0.8, feature sampling 0.6, and 919 boosting rounds at the best iteration. In the separate imbalance experiment with the initial XGBoost settings, PR-AUC was 0.855215 without weighting, 0.853195 with class weighting, and 0.811796 with SMOTE applied to training alone. Reusing validation for early stopping and model selection can make these scores optimistic.

The last cell saves the selected estimator, its fitted preprocessor, the expected feature order, all candidates' validation scores, the winner's validation metrics, and the package versions in `models/fraud_model.joblib`. It reloads the artifact and checks that it reproduces every validation score. The artifact is generated locally and excluded from Git. To score a feature frame returned by `src.data.xy` for Step 6, load the artifact from the repository root and reuse its preprocessor.

```python
import joblib

artifact = joblib.load("models/fraud_model.joblib")
X = artifact["preprocessor"].transform(X_raw)[artifact["feature_columns"]]
model = artifact["model"]
if artifact["name"] == "isolation_forest":
    scores = -model.score_samples(X)
else:
    scores = model.predict_proba(X)[:, 1]
```


## Step 6 · Evaluate the selected model at a cost-based threshold

Run [notebooks/04_evaluation_shap.ipynb](notebooks/04_evaluation_shap.ipynb) after [Step 5](#step-5--compare-and-select-fraud-models). The notebook reloads the selected model and its training-fitted preprocessor, then verifies that it reproduces the saved validation scores. If the model artifact predates the saved validation scores, rerun Step 5.

The simulation charges the original transaction amount for each missed fraud and a fixed fee per false alert. It treats the amounts as USD and assumes that flagging a fraud prevents the full loss, with no added cost for true alerts. [The cost functions](src/evaluate.py) scan all distinct score cutoffs on validation, including a policy with no alerts. Equal scores stay together and equal costs favor fewer alerts. Each selected cutoff is applied to the same chronological test scores.

These results are from macOS with Python 3.12.10, XGBoost 3.4.1, pandas 3.1.1, and scikit-learn 1.9.1. At the main USD 5 fee, the fixed cutoff of **0.574390531** produced 39 true alerts, 14 false alerts, and 13 missed frauds among 42,561 test transactions. Precision was **73.58%** and recall was **75.00%**. The test PR-AUC was **0.765433**, with a 95% percentile bootstrap interval of **[0.653280, 0.865480]** from 1,000 replicates. Test ROC-AUC was 0.980369.

The main test scenario cost **USD 2,442.40** (USD 2,372.40 in missed fraud and USD 70.00 in false alerts), versus **USD 6,168.88** without alerts. The simulated savings were **USD 3,726.48 (60.41%)**.

| Fee per false alert (USD) | Validation cutoff | Validation cost (USD) | Test cost (USD) | Test savings |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.574390531 | 182.30 | 2,400.40 | 61.09% |
| 5 | 0.574390531 | 206.30 | 2,442.40 | 60.41% |
| 10 | 0.574390531 | 246.30 | 2,512.40 | 59.27% |

The three fees selected the same cutoff, so the alert decisions stayed fixed as the fee changed. The missed frauds accounted for 38.5% of the fraudulent amount in test. Their median amount was USD 12.31 versus USD 4.69 for detected frauds. Four of the 13 misses occurred at relative hour 19. V14 had the largest median gap between missed and detected cases among the three leading features by validation permutation importance.

[Validation precision–recall curves](reports/figures/05_pr_curves.png) · [Validation cost by cutoff](reports/figures/06_cost_vs_threshold.png) · [Test confusion matrix](reports/figures/07_confusion_test.png)

The dataset spans about two days, test contains 52 frauds, and the PCA components are anonymized. The simulated cost depends on the fee and the assumed fraud loss. The bootstrap resamples individual transactions from test using fixed model scores. Model selection and threshold selection used validation. Step 4 reported baseline metrics on the same test split.

Run `python -m pytest -q` to check the cost calculation, tied scores and the policies of alerting on every transaction or on none.

## Step 7 · Explain the selected XGBoost model with SHAP

Continue through the SHAP section of [notebooks/04_evaluation_shap.ipynb](notebooks/04_evaluation_shap.ipynb). It reuses the Step 5 model and preprocessor and the Step 6 threshold. The sample contains all 52 test frauds and 2,000 randomly selected legitimate transactions (seed 42). Fraud is overrepresented in this sample, so the global ranking describes the sampled transactions.

TreeSHAP uses the 919 boosting rounds selected by early stopping. The baseline plus the feature contributions reconstructs the raw model margin in log-odds. Applying the logistic function recovered the scores used in Step 6 with a maximum absolute error of `2.51e-07` in the sample. The charts use SHAP 0.54.1. In the summary plot, red and blue indicate high and low feature values. Contributions to the right raise the fraud score and contributions to the left lower it.

`V4`, `V14`, and `V12` had the largest mean absolute contributions. Their median contributions raised the fraud score among fraudulent transactions and lowered it among legitimate transactions. `V14` showed the largest gap between the class medians of these three components. SHAP and XGBoost gain agreed on the top five features as a set, with a Spearman rank correlation of 0.652 across all 29 features. SHAP ranked `V4` first and gain ranked `V14` first. SHAP measures contributions to predictions in this sample, while gain measures average improvement to the training objective from splits on a feature. Features with more candidate splits can receive higher gain.

The notebook explains the largest fraud by original amount in each test outcome at the fixed cutoff of **0.574391**.

| Test fraud | Amount (USD) | Score | Largest signed contributions (log-odds) |
| --- | ---: | ---: | --- |
| [Detected](reports/figures/10_shap_waterfall_fraud.png) | 1,504.93 | 0.998774 | `V14` +2.762, `V10` +1.468, `V4` +0.944 |
| [Missed](reports/figures/11_shap_waterfall_missed_fraud.png) | 1,096.99 | 0.001837 | `V14` −1.891, `V12` −1.624, `V4` −0.948 |

[SHAP summary plot](reports/figures/08_shap_summary.png) · [Mean absolute SHAP contributions](reports/figures/09_shap_bar.png)

`V1` through `V28` are anonymized PCA components with unknown business meanings. SHAP identifies which components drive each model score and in which direction. Business variables such as merchant country, card type, or recent transaction frequency would give analysts more context for interpreting the contributions.

## Step 8 · Reproduce training and run the tests

[![tests](https://github.com/luis19vallejo02/fraud-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/luis19vallejo02/fraud-detection/actions/workflows/tests.yml)

With the Python 3.12 environment from Step 0 and `data/raw/creditcard.csv` from Step 1, run the following from the repository root:

```bash
python -m src.train
pytest -q
```

[`config.yaml`](config.yaml) records the Step 5 winning hyperparameters, random seed, split proportions, false-alert cost, and output paths. The script fits preprocessing and class weighting on training data, uses validation for early stopping and threshold selection, then evaluates the fixed model and cutoff on the chronological test set. Pass `--config path/to/config.yaml` to use a different configuration. Relative data and output paths resolve from the configuration file's directory.

The command writes `models/xgb_model.joblib` with the fitted estimator, preprocessor, expected feature order, and threshold. It also writes [`reports/metrics.json`](reports/metrics.json) with the validation and test results. A `null` threshold in JSON denotes the no-alert policy, stored as positive infinity in the model bundle. The model bundle stays local, and the notebooks can still use their earlier `models/fraud_model.joblib` artifact.

Two runs on the original CSV in the documented macOS environment produced identical metrics. The run selected 919 boosting rounds and reproduced the Step 6 cutoff of 0.574390531, test average precision of 0.765433, and simulated test cost of USD 2,442.40. The automated tests use synthetic data to check time boundaries, training-only preprocessing, recall, cost, and two executions of the training command. GitHub Actions runs the same suite on Python 3.12 without downloading the dataset.

## Step 9 · Serve predictions and report on alerts

After Step 8 creates `models/xgb_model.joblib`, launch the API from the repository root:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @sample_request.json
```

Open [Swagger UI](http://127.0.0.1:8000/docs) to send requests interactively. `POST /predict` accepts 28 finite PCA components in `V1`–`V28` order and a finite, nonnegative `amount` in the original units. It applies the training-fitted preprocessor and returns the uncalibrated XGBoost `fraud_score`, the `is_fraud_alert` decision, and the validation-selected `threshold`. A score at the threshold triggers an alert. A `null` threshold denotes a no-alert policy. [`sample_request.json`](sample_request.json) is a held-out row from the public dataset that triggers an alert with the documented model. Set `FRAUD_MODEL_PATH` to serve another trusted bundle produced by the same training pipeline.

Build and run the container after training, from the same directory:

```bash
docker build -t fraud-api .
docker run --rm -p 127.0.0.1:8000:8000 fraud-api
```

The build context includes the API, its dependencies and the local trained bundle. The [Docker workflow](.github/workflows/docker.yml) trains on synthetic transactions and checks `/health`, `/docs`, and `/predict` in a running container.

Use the saved model to regenerate the chronological test predictions and dashboard data:

```bash
python -m src.export_predictions
```

`reports/test_predictions.csv` contains row-level predictions and stays local. [`reports/dashboard_hourly.csv`](reports/dashboard_hourly.csv) contains the aggregates from the public dataset used in the [Data Studio dashboard](https://datastudio.google.com/reporting/e26dfb4e-32dd-4725-8df1-1e14a1c0de4d/page/DQIEG). After regenerating the export, replace the values in the connected Google Sheet with this aggregate CSV, keeping the header row and field names. Refresh the report data to load the new totals.

Use the **Hora relativa** selector to filter both charts and all six indicators. Hours count from the first transaction in the dataset. The report compares total and true alerts by hour and shows how the confirmed fraud amount splits between flagged and missed transactions. Alert precision uses `SUM(true_alerts) / SUM(alerts)`. Fraud amount coverage uses `SUM(detected_fraud_amount) / NULLIF(SUM(detected_fraud_amount) + SUM(missed_fraud_amount), 0)`. Both ratios are computed after filtering. The detected amount is the sum of amounts on flagged, confirmed frauds in the evaluation dataset. It represents simulated detection.

| Test transactions | Alerts | True alerts | Alert precision | Detected fraud amount | Missed fraud amount | Fraud amount coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42,561 | 53 | 39 | 73.58% | USD 3,796.48 | USD 2,372.40 | 61.54% |
