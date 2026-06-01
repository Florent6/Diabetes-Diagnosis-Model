import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import base64

#styles
st.set_page_config(page_title="🩺 Diabetes Risk Predictor", layout="wide")

def local_css():
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(to bottom right, #0c1e2c, #1f3b4d);
                color: #f0f0f0;
                font-family: 'Segoe UI', sans-serif;
            }
            h1, h2, h3, h4, h5, .stMarkdown p, label, legend {
                color: #f5f5f5 !important;
            }
            .stNumberInput > div > div > input, .stSelectbox div[data-baseweb="select"] {
                background-color: #f0f0f0 !important;
                color: #000 !important;
                border-radius: 10px;
                padding: 8px;
                border: none;
            }
            .stNumberInput input:focus {
                border: 1px solid #4dd0e1 !important;
            }
            button[kind="primary"] {
                background-color: #00c6ff;
                color: #ffffff;
                font-weight: bold;
                padding: 0.7em 1.4em;
                border-radius: 12px;
                border: none;
                box-shadow: 0 0 10px rgba(0,198,255,0.6);
                transition: 0.2s ease-in-out;
            }
            button[kind="primary"]:hover {
                background-color: #00b0e6;
                box-shadow: 0 0 20px rgba(0,198,255,0.8);
            }
        </style>
    """, unsafe_allow_html=True)

local_css()

#load and train model

df = pd.read_csv("dataset/diabetes_cleaned.csv")

X = df.drop(columns=["diabetes"])
y = df["diabetes"]

categorical = X.select_dtypes(include="object").columns.tolist()
numerical = [col for col in X.columns if col not in categorical]

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(drop='first'), categorical)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

#ui header
st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter patient information to estimate their risk of diabetes using machine learning.")

#forms
st.subheader("📋 Patient Information")

user_input = {}
col1, col2 = st.columns(2)

for i, col in enumerate(numerical):
    with (col1 if i % 2 == 0 else col2):
        user_input[col] = st.number_input(col.replace("_", " ").capitalize(), value=float(X[col].mean()))

for cat in categorical:
    label = "Gender" if cat.lower() == "gender" else cat.replace("_", " ").capitalize()
    user_input[cat] = st.selectbox(label, sorted(df[cat].unique()))

#predict
if st.button("🔍 Predict Diabetes Risk"):
    st.subheader("🔮 Result")
    input_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High likelihood of having diabetes.\n\n**Model confidence: {round(prob * 100, 2)}%**")
    else:
        st.success(f"✅ Low likelihood of having diabetes.\n\n**Model confidence: {round((1 - prob) * 100, 2)}%**")

#performance (real test)
with st.expander("📈 Model Performance"):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    st.markdown(f"**Accuracy:** {round(acc * 100, 2)}%")
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
