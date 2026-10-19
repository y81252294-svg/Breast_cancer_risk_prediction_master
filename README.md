# Breast-cancer-risk-prediction

This project uses the Wisconsin Breast Cancer dataset to train a machine learning model that predicts whether a tumor is malignant or benign.

## Overview
The repository is a learning project focused on:
- exploratory data analysis
- preprocessing and feature preparation
- support vector machine classification
- evaluation of model performance

## Project structure
- `data/data.csv` — dataset
- `train_model.py` — executable model training script
- `requirements.txt` — Python dependencies
- `NB1_IdentifyProblem+DataClean.ipynb` — problem definition and data loading
- `NB2_ExploratoryDataAnalysis.ipynb` — exploratory analysis
- `NB3_DataPreprocesing.ipynb` — preprocessing
- `NB4_PredictiveModelUsingSVM.ipynb` — SVM model
- `NB5 OptimizingSVMClassifier.ipynb` — SVM tuning
- `NB6 Comparison between different classifiers.ipynb` — model comparison

## Setup

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python train_model.py
   ```

## Run the interactive app

Run the Streamlit interface from the repository root:

```bash
streamlit run app.py
```

The app provides tumor-measurement inputs, benign/malignant predictions, probability estimates, and a small held-out accuracy summary. It is for educational use only and must not be used as a medical diagnosis.

## Run the web API locally

Start the FastAPI app with:

```bash
uvicorn api.index:app --reload
```

Open `http://127.0.0.1:8000/` for the browser interface. The prediction endpoint is `POST http://127.0.0.1:8000/api/predict` and requires all 30 numeric tumor measurements; values must be finite and non-negative. FastAPI's interactive API docs are available at `http://127.0.0.1:8000/docs`.

The included `vercel.json` configures the FastAPI app for Vercel deployment. The Streamlit interface is intended to be run separately with `streamlit run app.py`.
`pyproject.toml` contains only the FastAPI deployment dependencies so Vercel does not bundle the local Streamlit and visualization stack. `requirements.txt` retains the complete dependencies for running the Streamlit app, notebooks, and training script locally.

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fy81252294-svg%2FBreast_cancer_risk_prediction_master)

1. Select **Deploy with Vercel** above and sign in to Vercel.
2. Import `y81252294-svg/Breast_cancer_risk_prediction_master`. Keep the project root set to the repository root.
3. Leave the build and output settings at their defaults, then select **Deploy**.
4. After deployment completes, open the assigned Vercel URL. The home page serves the FastAPI prediction interface; `/docs` serves the API documentation.

When deploying through the GitHub integration, later pushes to `main` trigger new Vercel deployments automatically. The Streamlit interface is for local use and is not part of the Vercel deployment.

## What the model does
The script:
- loads the breast cancer CSV file
- removes the non-predictive ID column
- splits the data into training and testing sets
- scales the feature values
- trains a Support Vector Machine classifier
- prints the accuracy, confusion matrix, and classification report

## Expected result
A working SVM classifier should produce strong performance on the Wisconsin breast cancer dataset, typically with high accuracy and strong sensitivity/specificity.
