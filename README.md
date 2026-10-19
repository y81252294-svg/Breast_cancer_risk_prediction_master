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

