from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'data.csv'


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def load_dataset():
    df = load_data()
    df = df.drop(columns=['id'], errors='ignore')
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis'].map({'B': 0, 'M': 1})
    return X, y


def train_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = CalibratedClassifierCV(
        SVC(kernel='rbf', C=1.0, gamma='scale'),
        ensemble=False,
    )
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))

    return scaler, model, acc, cm


FEATURE_COLUMNS = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]


@st.cache_resource
def get_model():
    return train_model()


st.set_page_config(page_title='Breast Cancer Predictor', page_icon='🩺', layout='wide')

st.markdown(
    """
    <style>
    .main {
        background: #000000;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 0.7rem 0.9rem;
        box-shadow: none;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background: #000000 !important;
        color: #ffffff !important;
    }
    h1, h2, h3, h4, p, label, div, span {
        color: #ffffff !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stButton > button {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #ffffff !important;
        color: #000000 !important;
    }
    .stAlert, .stSuccess, .stError, .stWarning, .stInfo {
        background: #111111 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('🩺 Breast Cancer Risk Prediction')
st.markdown('<p style="font-size:1.08rem; color:#e5e7eb;">A lightweight machine learning demo that estimates whether a breast tumor is likely <b>benign</b> or <b>malignant</b>.</p>', unsafe_allow_html=True)
st.warning('Educational demonstration only; this prediction is not a diagnosis or a substitute for professional medical advice.')

scaler, model, accuracy, cm = get_model()

col1, col2, col3 = st.columns(3)
col1.metric('Model Accuracy', f'{accuracy * 100:.2f}%')
col2.metric('Benign Correct', int(cm[0][0]))
col3.metric('Malignant Correct', int(cm[1][1]))

st.subheader('Enter tumor measurements')

data = load_data()
sample = data.iloc[0].to_dict()
example_button = st.button('Use sample values')

input_values = {}
for feature in FEATURE_COLUMNS:
    default_value = float(sample.get(feature, 0.0))
    if feature == 'diagnosis':
        continue
    input_values[feature] = st.number_input(
        feature.replace('_', ' ').title(),
        min_value=0.0,
        value=default_value if example_button else float(data[feature].median()),
        step=0.01,
        format='%.4f'
    )

if st.button('Predict Diagnosis', type='primary'):
    row = pd.DataFrame([input_values], columns=FEATURE_COLUMNS, dtype=float)
    scaled_row = scaler.transform(row)
    prediction = model.predict(scaled_row)[0]
    label = 'Malignant' if prediction == 1 else 'Benign'

    proba = model.predict_proba(scaled_row)[0]
    benign_prob = proba[0] * 100
    malignant_prob = proba[1] * 100

    st.subheader('Result')
    if prediction == 1:
        st.error(f'Prediction: {label}')
        st.warning(f'Malignant probability: {malignant_prob:.2f}%')
        st.info('This case is predicted as malignant and should be reviewed by a medical professional.')
    else:
        st.success(f'Prediction: {label}')
        st.info(f'Benign probability: {benign_prob:.2f}%')
        st.success('This case is predicted as benign based on the current input values.')

st.caption('Dataset source: Wisconsin Breast Cancer Dataset')
