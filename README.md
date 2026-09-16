# MLOps First Project

An end-to-end machine learning project built with Python, Scikit-learn, ZenML, MLflow, Optuna, Pytest, and GitHub Actions.

The main goal of this project is not only to train a machine learning model, but to build a reproducible and testable ML pipeline with data preprocessing, model training, hyperparameter optimization, experiment tracking, automated testing, and continuous integration.

---

## 📌 Project Overview

This project uses the **Spaceship Titanic** dataset to build a binary classification system that predicts whether a passenger was transported to another dimension.

Instead of implementing the project as a single notebook, the machine learning workflow is organized into modular components and ZenML pipeline steps.

The project includes:

- Data ingestion
- Data preprocessing
- Feature engineering
- Train / validation / test splitting
- Multiple classification models
- Hyperparameter optimization with Optuna
- Model evaluation
- MLflow metric logging
- ZenML pipeline orchestration
- Unit testing with Pytest
- Continuous Integration with GitHub Actions

---

## 🎯 Problem

The objective is to predict the `Transported` target variable from passenger information.

The dataset contains information such as:

- Home Planet
- CryoSleep
- Cabin
- Destination
- Age
- VIP status
- Room Service
- Food Court
- Shopping Mall
- Spa
- VR Deck

The target variable is:

```text
Transported
```

which represents whether a passenger was transported.

---

## 🏗️ Project Architecture

The overall machine learning workflow is organized as follows:

```text
                    Raw CSV Data
                         │
                         ▼
                  Data Ingestion
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
              Feature Engineering
                         │
                         ▼
              Train / Validation /
                    Test Split
                         │
                         ▼
              Scikit-learn Pipeline
                         │
                         ▼
                  Model Training
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        Random Forest  XGBoost   LightGBM
              │          │          │
              └──────────┼──────────┘
                         │
                         ▼
              Hyperparameter Tuning
                    (Optuna)
                         │
                         ▼
                   Model Evaluation
                         │
                         ▼
                      MLflow
                         │
                         ▼
                    Test Results
                         │
                         ▼
                  GitHub Actions
```

---

## 🧰 Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and machine learning |
| XGBoost | Gradient boosting model |
| LightGBM | Gradient boosting model |
| CatBoost | Machine learning dependency |
| Optuna | Hyperparameter optimization |
| ZenML | Pipeline orchestration |
| MLflow | Experiment tracking and metric logging |
| Pytest | Automated testing |
| Git | Version control |
| GitHub | Source code hosting |
| GitHub Actions | Continuous Integration |

---

## 📁 Project Structure

```text
First Project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── Space_train.csv
│   └── Space_test.csv
│
├── pipelines/
│   └── training_pipeline.py
│
├── saved_model/
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── evaluation.py
│   └── model_dev.py
│
├── steps/
│   ├── __init__.py
│   ├── clean_data.py
│   ├── config.py
│   ├── evaluation.py
│   ├── ingest_data.py
│   └── model_train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_clean_data.py
│   ├── test_data_cleaning.py
│   ├── test_ingest_data.py
│   ├── test_model_dev.py
│   ├── test_model_train.py
│   ├── test_src_evaliation.py
│   └── test_steps_evaluation.py
│
├── .gitignore
├── pytest.ini
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

## 🔄 Data Pipeline

The data processing workflow is divided into separate stages.

### 1. Data Ingestion

The `ingest_df` ZenML step reads the dataset from a CSV file.

```text
CSV
 ↓
Pandas DataFrame
```

The ingestion logic is implemented in:

```text
steps/ingest_data.py
```

---

### 2. Data Preprocessing

The preprocessing logic is implemented in:

```text
src/data_cleaning.py
```

The preprocessing stage performs operations such as:

- Removing unnecessary columns
- Splitting the `Cabin` feature into separate features
- Handling missing values
- Converting numerical values
- Encoding categorical features
- Preparing the data for machine learning

For example:

```text
Cabin
  │
  ├── Deck
  ├── Num
  └── Side
```

---

### 3. Data Splitting

The dataset is divided into:

```text
Training Set
Validation Set
Test Set
```

Stratified splitting is used to preserve the class distribution.

---

### 4. Feature Transformation

The project uses Scikit-learn preprocessing components including:

- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`
- `PolynomialFeatures`

The preprocessing pipeline is fitted using the training data and then applied to validation and test data.

This keeps the preprocessing workflow consistent between training and evaluation.

---

## 🤖 Machine Learning Models

The project currently provides several classification models:

### Random Forest

Implemented through:

```text
RandomForestModel
```

### XGBoost

Implemented through:

```text
XGBModel
```

### LightGBM

Implemented through:

```text
LightGBMModel
```

### Logistic Regression

Implemented through:

```text
LogisticRegressionModel
```

All models follow the common `Model` abstraction defined in:

```text
src/model_dev.py
```

This makes it possible to switch between models without changing the overall training pipeline.

---

## 🔬 Hyperparameter Optimization

Hyperparameter optimization is handled by **Optuna**.

The project contains a:

```text
HyperParameterTuner
```

which creates an Optuna study and searches for better hyperparameter configurations.

For example, Random Forest optimization includes parameters such as:

```text
n_estimators
max_depth
min_samples_split
```

The optimization objective is based on the model's validation performance.

---

## 📊 Model Evaluation

The evaluation stage calculates classification metrics including:

- Accuracy
- Precision
- Recall
- F1 Score

The evaluation step is implemented in:

```text
steps/evaluation.py
```

The calculated metrics are also logged to MLflow.

---

## 🧪 Testing

The project uses **Pytest** for automated unit testing.

The tests cover multiple parts of the application:

```text
tests/
│
├── test_data_cleaning.py
├── test_clean_data.py
├── test_ingest_data.py
├── test_model_dev.py
├── test_model_train.py
├── test_src_evaliation.py
└── test_steps_evaluation.py
```

The tests cover areas including:

- Data preprocessing
- Data splitting
- Feature transformation
- Model creation
- Model training
- Hyperparameter optimization
- Data ingestion
- ZenML steps
- Model evaluation
- MLflow metric logging

Run the complete test suite with:

```bash
pytest -v
```

---

## 🔄 ZenML Pipeline

The machine learning workflow is organized using ZenML.

The main steps are:

```text
ingest_df
    ↓
clean_df
    ↓
train_model
    ↓
evaluate_model
```

The pipeline definition is located in:

```text
pipelines/training_pipeline.py
```

This separates individual ML operations into reusable pipeline steps.

---

## 📈 MLflow

MLflow is used for experiment tracking and metric logging.

The evaluation step logs:

```text
accuracy
precision
recall
f1_score
```

This makes it possible to track model evaluation results without embedding experiment tracking directly into the model implementation.

---

## ⚙️ Continuous Integration

The project uses **GitHub Actions** for Continuous Integration.

The workflow is located at:

```text
.github/workflows/ci.yml
```

The CI workflow automatically checks the project after changes are pushed to GitHub.

The general workflow is:

```text
Git Push
   │
   ▼
GitHub Actions
   │
   ▼
Set up Python
   │
   ▼
Install Dependencies
   │
   ▼
Run Tests
   │
   ▼
Check Project
```

This helps ensure that changes do not silently break existing functionality.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd First-Project
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the Tests

After installing the dependencies:

```powershell
pytest -v
```

---

## ▶️ Running the Pipeline

The training pipeline can be started through:

```powershell
python run_pipeline.py
```

The pipeline then executes the configured ZenML workflow.

---

## 🧩 Configuration

Model configuration is handled through:

```text
steps/config.py
```

The project supports selecting the model and controlling whether hyperparameter tuning is enabled.

Example configuration:

```python
ModelNameConfig(
    model_name="random_forest",
    fine_tuning=True
)
```

---

## 🛡️ Reproducibility

The project uses several practices to make the workflow more reproducible:

- Fixed random seeds where appropriate
- Version-pinned dependencies
- Separate preprocessing and training stages
- Automated tests
- Git version control
- CI checks through GitHub Actions
- Pipeline orchestration through ZenML

---

## 📌 Current Project Status

### Completed

- [x] Project structure
- [x] Data ingestion
- [x] Data preprocessing
- [x] Feature engineering
- [x] Train / validation / test split
- [x] Multiple machine learning models
- [x] Hyperparameter optimization
- [x] Model evaluation
- [x] ZenML pipeline
- [x] MLflow integration
- [x] Pytest unit tests
- [x] GitHub Actions CI

### Planned

- [ ] Containerization with Docker
- [ ] Model serving / deployment
- [ ] Additional production-oriented improvements

---

## 🎯 Learning Goals

This project was developed to move beyond notebook-based machine learning and practice the complete development workflow of an ML project.

The main learning goals are:

- Writing modular Python code
- Building reproducible ML pipelines
- Applying software engineering practices to machine learning
- Writing automated tests
- Using experiment tracking
- Using pipeline orchestration
- Implementing Continuous Integration
- Structuring a project for GitHub
- Preparing an ML system for future deployment

---

## 📄 License

This project is currently intended as a personal learning and portfolio project.
