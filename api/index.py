from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import ConfigDict, Field, create_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'data.csv'
FEATURE_COLUMNS = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

PredictRequest = create_model(
    'PredictRequest',
    __config__=ConfigDict(extra='forbid'),
    **{
        feature: (float, Field(ge=0, allow_inf_nan=False))
        for feature in FEATURE_COLUMNS
    },
)

app = FastAPI(title='Breast Cancer Risk Prediction')


@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv(DATA_PATH).drop(columns=['id'], errors='ignore')


@lru_cache(maxsize=1)
def get_model():
    df = load_data()

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis'].map({'B': 0, 'M': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    _ = scaler.transform(X_test)

    model = CalibratedClassifierCV(
        SVC(kernel='rbf', C=1.0, gamma='scale'),
        ensemble=False,
    )
    model.fit(X_train_scaled, y_train)
    return scaler, model


@app.get('/', response_class=HTMLResponse)
async def home():
    sample = load_data()[FEATURE_COLUMNS].median().to_dict()
    field_html = ''.join(
        f'<div class="field"><label for="{feature}">{feature}</label><input id="{feature}" name="{feature}" type="number" min="0" step="any" required value="{sample[feature]}" /></div>'
        for feature in FEATURE_COLUMNS
    )
    features_json = str(FEATURE_COLUMNS).replace("'", '"')

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Breast Cancer Risk Prediction</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #000000;
                margin: 0;
                padding: 32px;
                color: #ffffff;
            }}
            .container {{
                max-width: 960px;
                margin: auto;
                background: #111111;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 10px 30px rgba(255,255,255,0.04);
            }}
            h1 {{ margin-top: 0; color: #ffffff; }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
            }}
            .field {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                margin-bottom: 8px;
            }}
            label {{ color: #e5e7eb; }}
            input {{
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.2);
                background: #0a0a0a;
                color: #ffffff;
                font-size: 1rem;
            }}
            button {{
                background: #ffffff;
                color: #000000;
                border: none;
                border-radius: 10px;
                padding: 12px 18px;
                font-size: 1rem;
                cursor: pointer;
                margin-top: 16px;
                font-weight: 700;
            }}
            .result {{
                margin-top: 24px;
                padding: 18px;
                border-radius: 12px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.1);
                font-size: 1.05rem;
                color: #ffffff;
                white-space: pre-line;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                background: rgba(255,255,255,0.1);
                color: #ffffff;
                font-size: 0.8rem;
                font-weight: bold;
                border: 1px solid rgba(255,255,255,0.12);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🩺 Breast Cancer Risk Prediction</h1>
            <p>
                This educational demo uses an SVM classifier to estimate whether a tumor sample is
                <strong>benign</strong> or <strong>malignant</strong>.
            </p>
            <p>This is not a diagnosis or a substitute for professional medical advice.</p>
            <p><span class="badge">Demo model</span></p>

            <form id="predict-form">
                <div class="grid">
                    {field_html}
                </div>
                <button type="submit">Predict Diagnosis</button>
            </form>

            <div id="result" class="result">Prediction: waiting for input...</div>
        </div>

        <script>
            const features = {features_json};
            const form = document.getElementById('predict-form');
            const result = document.getElementById('result');

            form.addEventListener('submit', async (event) => {{
                event.preventDefault();
                const payload = {{}};
                for (const feature of features) {{
                    const value = Number(document.getElementById(feature).value);
                    payload[feature] = value;
                }}

                try {{
                    const response = await fetch('/api/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(payload)
                    }});

                    const data = await response.json();
                    if (!response.ok) {{
                        const detail = Array.isArray(data.detail)
                            ? data.detail.map((error) => error.msg).join(', ')
                            : data.detail || 'The request was rejected.';
                        result.textContent = `Prediction failed: ${{detail}}`;
                        return;
                    }}

                    result.textContent = `Prediction: ${{data.prediction}}
Benign probability: ${{data.benign_probability.toFixed(2)}}%
Malignant probability: ${{data.malignant_probability.toFixed(2)}}%`;
                }} catch (error) {{
                    result.textContent = 'Prediction request failed. Check the connection and try again.';
                }}
            }});
        </script>
    </body>
    </html>
    """


@app.post('/api/predict')
def predict(payload: PredictRequest):
    scaler, model = get_model()
    row = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS, dtype=float)
    scaled_row = scaler.transform(row)
    prediction = int(model.predict(scaled_row)[0])
    probabilities = model.predict_proba(scaled_row)[0]

    benign_probability = float(probabilities[0] * 100)
    malignant_probability = float(probabilities[1] * 100)
    diagnosis = 'Malignant' if prediction == 1 else 'Benign'

    return {
        'prediction': diagnosis,
        'benign_probability': round(benign_probability, 2),
        'malignant_probability': round(malignant_probability, 2),
        'model_status': 'ready'
    }
