# ML Projects

A single repo hosting three self-contained machine-learning projects, each in
its own folder:

- **`intent-anlaysis/`** — customer-support **intent** classifier (77 banking intents).
- **`customer-review-analysis/`** — product-review **sentiment** classifier (positive / negative).
- **`real-estate-regression/`** — California **house-price** regressor, solved
  with both a random forest and a TensorFlow/Keras neural network.

Each project trains a model end-to-end and demonstrates data preparation,
training, evaluation and inference inside Jupyter notebooks.

## What's in this repo

```
.
├── intent-anlaysis/
│   ├── intent_analysis.ipynb       # Intent workflow + demo (main artifact)
│   ├── train.csv                   # Training dataset
│   ├── test.csv                    # Test dataset
│   ├── ticket_intent_model.joblib  # Trained model (generated)
│   └── validation_errors.csv       # Error-analysis output (generated)
├── customer-review-analysis/
│   ├── customer_review.ipynb           # Review workflow + demo (main artifact)
│   ├── flipkart_data.csv               # Raw reviews
│   ├── flipkart_data_with_sentiment.csv
│   ├── flipkart_data_binary.csv
│   └── customer_review_model.joblib    # Trained model (generated)
├── real-estate-regression/
│   ├── real-estate-regression.ipynb    # Random-forest workflow (main artifact)
│   ├── real-esate-neural-network.ipynb # Neural-network workflow (TensorFlow/Keras)
│   ├── california_housing_rf.joblib    # Trained forest (generated)
│   └── california_housing_nn.keras     # Trained neural network (generated)
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project: Intent analysis (`intent-anlaysis/`)

Maps a free-text support message to one of **77 banking support intents** using
classic NLP (TF-IDF) plus Logistic Regression, trained on the
[PolyAI banking support dataset](https://github.com/PolyAI-LDN/task-specific-datasets).

Training and the demo live in `intent_analysis.ipynb`.

`ticket_intent_model.joblib` and `validation_errors.csv` are generated when you
run the notebook.

### Features

- End-to-end notebook: loading, analysis, training, tuning and evaluation.
- Text pipeline: `TfidfVectorizer` (n-grams) → `LogisticRegression`.
- Model comparison (Logistic Regression vs. Decision Tree).
- Feature (n-gram) and regularization (`C`) experiments.
- Error analysis with saved mis-classifications.
- Trained model persisted with `joblib`.
- Interactive demo: predict an intent and explain which words drive it.

### Model

| Stage      | Component                                       |
| ---------- | ----------------------------------------------- |
| Features   | `TfidfVectorizer(lowercase=True, ngram=(1,1))`  |
| Classifier | `LogisticRegression(C=10.0, max_iter=1000)`     |
| Task       | Multi-class text classification (77 intents)    |

Evaluation uses **accuracy** and **macro-F1** to account for class imbalance.

### Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook intent-anlaysis/intent_analysis.ipynb
```

- Sections 1–9 train and save the model to `ticket_intent_model.joblib`.
- Section 10 is the interactive demo: set the `text` variable to any customer
  message to see the predicted intent plus the words that drove the decision.

---

## Project: Customer review (`customer-review-analysis/`)

Maps the free-text of a product review to a **positive** or **negative**
sentiment using classic NLP (TF-IDF) plus Logistic Regression, trained on the
Flipkart product reviews dataset. A three-way (positive / neutral / negative)
label is derived from the star rating, then neutral reviews are dropped to build
the binary training set.

Training and the demo live in `customer_review.ipynb`.

`flipkart_data_with_sentiment.csv`, `flipkart_data_binary.csv` and
`customer_review_model.joblib` are generated when you run the notebook.

### Features

- End-to-end notebook: loading, sentiment labelling, training and evaluation.
- Text pipeline: `TfidfVectorizer` (n-grams) → `LogisticRegression`.
- Model comparison (Logistic Regression vs. Naive Bayes vs. Decision Tree).
- Regularization (`C`), smoothing (`alpha`) and n-gram experiments.
- Threshold analysis on the negative class.
- Trained model persisted with `joblib`.
- Interactive demo: predict the sentiment of a single review.

### Model

| Stage      | Component                                                  |
| ---------- | ---------------------------------------------------------- |
| Features   | `TfidfVectorizer(lowercase=True, ngram=(1,1))`             |
| Classifier | `LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced')` |
| Task       | Binary text classification (positive / negative)           |

Evaluation uses **accuracy** and **macro-F1** to account for class imbalance.

### Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook customer-review-analysis/customer_review.ipynb
```

- Sections 1–8 explore the data and compare models.
- Section 9 trains and saves the model to `customer_review_model.joblib`.
- Section 10 is the interactive demo: set the `review` variable to any review to
  see the predicted sentiment and its confidence.

---

## Project: California housing regression (`real-estate-regression/`)

Predicts the **median house value** of a California census block group from
eight socio-economic and geographic features in the 1990 census, loaded with
scikit-learn's `fetch_california_housing`. The target (`MedHouseVal`) is
continuous and measured in **$100,000 units**.

The folder contains two complementary notebooks: this section covers the
random-forest workflow, and the next section covers the neural network.

The random-forest training, evaluation and inference live in
`real-estate-regression.ipynb`.

`california_housing_rf.joblib` is generated when you run the notebook.

### Features

- End-to-end notebook: EDA, baseline, model comparison, tuning and test evaluation.
- Baseline mean predictor to put model scores in context.
- Models compared: Linear Regression, Decision Tree, Ridge, Polynomial Ridge
  and Random Forest.
- Hyperparameter sweeps for tree depth, leaf size and regularisation strength.
- Error and residual analysis, including the worst predictions.
- Final random forest persisted with `joblib`, then reloaded for a sample
  prediction.

### Model

| Stage      | Component                                                       |
| ---------- | --------------------------------------------------------------- |
| Features   | 8 numeric census features (no scaling required for the forest)  |
| Estimator  | `RandomForestRegressor(n_estimators=100, min_samples_leaf=2)`   |
| Task       | Regression (median house value, $100,000 units)                 |

Evaluation uses **MAE**, **RMSE** and **R²**. The final random forest reaches
**MAE 0.343**, **RMSE 0.524** and **R² 0.800** on the held-out test set,
compared with **MAE 0.534 / R² 0.592** for linear regression and a mean
baseline MAE of **0.904**.

### Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook real-estate-regression/real-estate-regression.ipynb
```

- Sections 1–2 explore the data, target and outliers.
- Sections 3–4 build and diagnose a linear baseline.
- Section 5 compares tree, Ridge, polynomial and random forest models.
- Section 6 saves the final model and runs a sample prediction; Section 7
  summarises the results and suggests next steps.

---

## Project: California housing with a neural network (`real-estate-regression/`)

Solves the same regression task with a small feed-forward neural network built
in TensorFlow/Keras. It uses the same 60/20/20 split, target and metrics as the
random-forest notebook, so the two models can be compared directly.

Training, experiments and inference live in
`real-esate-neural-network.ipynb`.

`california_housing_nn.keras` is generated when you run the notebook.

### Features

- End-to-end notebook: loading, leakage-free normalization, training and test
  evaluation.
- Keras `Normalization` layer adapted on the training split only.
- MLP (8 → 64 → 32 → 1) with ReLU hidden layers and a linear output.
- Early stopping on validation loss with best-weight restoration.
- L2 regularization sweep (`lambda` = 0.0, 0.0001, 0.001).
- Residual analysis overall, by predicted price band, and as a scatter plot.
- Side-by-side comparison with a random forest on validation and test splits.
- Full model saved in the native `.keras` format and reloaded to verify
  identical predictions.

### Model

| Stage        | Component                                                              |
| ------------ | ---------------------------------------------------------------------- |
| Features     | 8 numeric census features, standardized by a Keras `Normalization` layer |
| Architecture | `Dense(64, relu)` → `Dense(32, relu)` → `Dense(1)`                     |
| Optimizer    | Adam (`learning_rate=0.001`)                                           |
| Loss / metric | MSE / MAE                                                             |
| Task         | Regression (median house value, $100,000 units)                        |

Evaluation uses **MAE**, **RMSE** and **R²**. With `lambda=0.0001`, the network
reaches **MAE 0.363**, **RMSE 0.534** and **R² 0.781** on the held-out test set,
compared with **MAE 0.341 / RMSE 0.516 / R² 0.796** for the random forest.

### Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook real-estate-regression/real-esate-neural-network.ipynb
```

- Sections 1–4 load the data, normalize it, build the MLP and train it with
  early stopping.
- Section 5 tunes L2 regularization; the best setting (`lambda=0.0001`) is
  refit for the final model.
- Section 6 analyses residuals.
- Sections 7–8 compare against a random forest and save/reload the Keras model.
- Section 9 summarises results and next steps.

## License

For educational use. Datasets © PolyAI-LDN, Flipkart and the California housing
dataset authors (StatLib / Pace & Barry), as distributed by scikit-learn.
