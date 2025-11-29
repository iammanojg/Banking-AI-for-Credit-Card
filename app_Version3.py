import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import joblib

# ---------- Page config ----------
st.set_page_config(page_title="Smart Spending Advisor", page_icon="💳", layout="centered")

st.title("💳 Smart Spending Advisor")
st.write("Enter a customer ID to see an analytical summary and an AI-driven recommendation.")

# ---------- Paths ----------
DATA_PATH = os.path.join("data", "spending_patterns_REALISTIC_97percent.csv")
MODEL_PATH = os.path.join("model", "catboost_model.joblib")

# ---------- Suggestions mapping (user-provided) ----------
SUGGESTIONS = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit!",
    'Debit Card': "Insight: While practical, a Credit Card could offer buyer protection and rewards for certain transactions.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Keep up the smart spending!",
    'Digital Wallet': "Smart move! For even more benefits, link your Digital Wallet to a rewards Credit Card."
}

# ---------- Utility: load CSV ----------
@st.cache_data(show_spinner=False)
def load_csv(maybe_uploaded):
    if maybe_uploaded is not None:
        df = pd.read_csv(maybe_uploaded)
        source = "uploaded file"
    elif os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        source = DATA_PATH
    else:
        # fallback small mock dataset (keeps app usable if user forgot to upload)
        df = pd.DataFrame({
            "Customer ID": ["CUST_0159", "CUST_0245", "CUST_0312"],
            "Transaction Date": pd.to_datetime(["2025-01-05", "2025-02-10", "2025-03-20"]),
            "Location": ["app", "store", "web"],
            "Category": ["Groceries", "Travel", "Fitness"],
            "Item": ["Milk", "Flight", "Gym"],
            "Quantity": [1, 1, 1],
            "Total Spent": [45.5, 280.0, 130.0],
            "Payment Method": ["Debit Card", "Credit Card", "Cash"]
        })
        source = "mock data"
    return df, source

# ---------- Feature engineering (same logic you used, adapted) ----------
def featurize(df):
    df = df.copy()
    # convert Transaction Date if present
    if 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
        df['Year'] = df['Transaction Date'].dt.year.fillna(0).astype(int)
        df['Month'] = df['Transaction Date'].dt.month.fillna(0).astype(int)
        df['Day'] = df['Transaction Date'].dt.day.fillna(0).astype(int)
        df['Weekday'] = df['Transaction Date'].dt.weekday.fillna(0).astype(int)
        df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
        df['Is_Month_End'] = df['Day'].isin([28,29,30,31]).astype(int)
        df.drop(columns=['Transaction Date'], inplace=True)
    # total spent / quantity
    if 'Total Spent' in df.columns:
        df['Total_Spent_Log'] = np.log1p(df['Total Spent'].fillna(0))
        df['Is_High_Value'] = (df['Total Spent'].fillna(0) > 500).astype(int)
        df['Is_Very_Cheap'] = (df['Total Spent'].fillna(0) < 10).astype(int)
    if 'Quantity' in df.columns:
        df['Price_Per_Unit'] = df['Total Spent'].fillna(0) / df['Quantity'].replace(0, 1).fillna(1)
    # channel extraction
    def get_channel(loc):
        loc = str(loc).lower()
        if any(x in loc for x in ['app','mobile','ios','android']): return 'Mobile App'
        if any(x in loc for x in ['web','online','site']): return 'Online'
        if any(x in loc for x in ['store','pos','shop','in-store']): return 'In-store'
        return 'Other'
    if 'Location' in df.columns:
        df['Channel'] = df['Location'].apply(get_channel)
        df.drop(columns=['Location'], inplace=True)
    # Ensure expected columns exist
    # Categorical candidates: Customer ID, Category, Item, Channel
    for c in ['Customer ID', 'Category', 'Item', 'Channel', 'Payment Method']:
        if c not in df.columns:
            df[c] = "Unknown"
    return df

# ---------- Model training/caching ----------
@st.cache_resource(show_spinner=False)
def train_model(full_df, iterations=300, depth=6, random_seed=42):
    df = featurize(full_df)
    # keep only relevant columns, drop rows with missing Payment Method
    df = df.dropna(subset=['Payment Method']).reset_index(drop=True)
    # split
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=random_seed, stratify=df['Payment Method'])
    X_train = train_df.drop(columns=['Payment Method'])
    y_train = train_df['Payment Method']
    X_test = test_df.drop(columns=['Payment Method'])
    y_test = test_df['Payment Method']

    # set categorical feature names for CatBoost (they can be string dtype)
    cat_features = [c for c in ['Customer ID', 'Category', 'Item', 'Channel'] if c in X_train.columns]

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        random_seed=random_seed,
        verbose=False
    )

    model.fit(X_train, y_train, cat_features=cat_features)
    # compute accuracy on holdout for debug
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return {
        "model": model,
        "cat_features": cat_features,
        "accuracy": acc,
        "X_columns": X_train.columns.tolist()
    }

# ---------- Sidebar: data upload / training options ----------
st.sidebar.header("Data & Model")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
df, source = load_csv(uploaded)
st.sidebar.markdown(f"**Data source:** {source} — {len(df):,} rows")

# training configuration
iterations = st.sidebar.number_input("CatBoost iterations", min_value=50, max_value=2000, value=300, step=50)
depth = st.sidebar.slider("Tree depth", 3, 12, 6)
retrain = st.sidebar.button("(Re)train model now")

# Train (cached)
with st.spinner("Training model (cached) — this may take a moment..."):
    model_bundle = train_model(df, iterations=int(iterations), depth=int(depth))
model = model_bundle["model"]

# ---------- Input & Analysis ----------
cust_id = st.text_input("Enter Customer ID (e.g., CUST_0159):").strip()

if st.button("🔍 Analyze Spending"):

    df_full = featurize(df)
    # find matching row(s)
    matches = df_full[df_full['Customer ID'].astype(str) == str(cust_id)]
    if matches.empty:
        st.error("Customer ID not found in the dataset. Try a different ID or upload a dataset with the desired customer.")
    else:
        user_row = matches.iloc[0]
        st.subheader(f"Customer Summary: {cust_id}")
        # Display a compact summary of important fields
        cols_to_show = [c for c in ['Category','Item','Total Spent','Quantity','Channel','Year','Month','Is_Weekend'] if c in user_row.index]
        for c in cols_to_show:
            st.write(f"**{c}:** {user_row[c]}")

        # Prepare X for prediction (use same columns as training)
        Xcols = model_bundle['X_columns']
        X_input = pd.DataFrame([user_row.reindex(Xcols)]).reset_index(drop=True)

        # Predict
        pred_raw = model.predict(X_input)
        # probability: model.predict_proba returns array of shape (n_samples, n_classes)
        proba = model.predict_proba(X_input)[0]
        # get class order
        class_order = model.classes_
        # predicted label and confidence
        pred_label = str(pred_raw[0])
        confidence = float(np.max(proba))

        st.success(f"🧠 Predicted Payment Method: **{pred_label}** (Confidence: {confidence:.2f})")

        # Analytical AI mock: small reasoning based on features
        analysis_msgs = []
        if 'Total Spent' in user_row.index:
            spent = float(user_row['Total Spent'])
            if spent > 500:
                analysis_msgs.append("High-value purchase — likely to benefit from card protections and rewards.")
            elif spent < 10:
                analysis_msgs.append("Small purchases — convenient methods like cash or digital wallets are common here.")
        if 'Channel' in user_row.index:
            ch = str(user_row['Channel'])
            if ch == 'Mobile App':
                analysis_msgs.append("This customer shops via Mobile App frequently — digital payment methods preferred.")
            elif ch == 'In-store':
                analysis_msgs.append("In-store purchases often correlate with cash or debit usage.")
        if analysis_msgs:
            st.markdown("### 🔎 Analytical AI")
            for m in analysis_msgs:
                st.write("- " + m)

        # Generative AI mock: personalized tip from your mapping
        gen_tip = SUGGESTIONS.get(pred_label, "Consider a Credit Card for rewards and protection.")
        st.markdown("### 💬 AI Recommendation")
        st.info(gen_tip)

        # Optional: feature importance (compact)
        with st.expander("Model details & diagnostics"):
            st.write(f"Model accuracy on holdout (debug): {model_bundle['accuracy']:.3f}")
            try:
                fi = model.get_feature_importance(type='FeatureImportance')
                names = model.feature_names_
                imp = sorted(zip(names, fi), key=lambda x: x[1], reverse=True)
                st.write("Top features:")
                for n, v in imp[:10]:
                    st.write(f"- {n}: {v:.3f}")
            except Exception as e:
                st.write("Feature importance not available:", e)

# ---------- Footer ----------
st.markdown("---")
st.caption("Built for quick trial. Keep sensitive datasets private — don't push them to public repos.")