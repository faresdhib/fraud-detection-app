# fraud_predictor_app.py

import numpy as np
import pandas as pd
import joblib
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os

# -----------------------------
# 1️⃣ Load the saved model & scaler
# -----------------------------
model = joblib.load("fraud_model.pkl")   # Trained Logistic Regression model
scaler = joblib.load("scaler.pkl")       # StandardScaler used for preprocessing

# -----------------------------
# 2️⃣ Core prediction (no UI stuff)
# -----------------------------
def _predict_core(features):
    """
    Internal: takes a 1D list/array of 30 features.
    Returns (result_dict, matplotlib_figure)
    """
    # Convert to 2D array
    features_array = np.array(features, dtype=float).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features_array)

    # Predict class and probabilities
    pred_class = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    normal_prob, fraud_prob = float(prob[0]), float(prob[1])

    # Build result dictionary
    result = {
        "🟢 Prediction": "✅ Normal Transaction" if pred_class == 0 else "🚨 Fraud Detected!",
        "📊 Probability Normal": f"{normal_prob*100:.2f}%",
        "📊 Probability Fraud": f"{fraud_prob*100:.2f}%"
    }

    # Create pie chart
    labels = ["Normal", "Fraud"]
    sizes = [normal_prob, fraud_prob]
    colors = ["#4CAF50", "#F44336"]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.axis("equal")
    plt.title("Fraud vs Normal Probability")

    return result, fig, pred_class, normal_prob, fraud_prob

# -----------------------------
# 3️⃣ Gradio logic helpers
# -----------------------------
def predict_and_log(*features, history):
    """
    Gradio-click handler:
    - runs prediction
    - appends a row to history (a list of dicts)
    - returns: result JSON, pie chart, history DataFrame, updated history state
    """
    try:
        result, fig, pred_class, normal_prob, fraud_prob = _predict_core(features)

        # Build a short preview of features for the table (first 5 + last 2)
        f = [float(x) for x in features]
        preview = f[:5] + ["…"] + f[-2:]

        row = {
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Prediction": "Fraud" if pred_class == 1 else "Normal",
            "Prob_Normal": round(normal_prob, 4),
            "Prob_Fraud": round(fraud_prob, 4),
            "Features_Preview": preview
        }

        history = history or []
        history.append(row)
        df_hist = pd.DataFrame(history)

        return result, fig, df_hist, history
    except Exception as e:
        # On error, keep history unchanged but surface the error message
        err_json = {"Error": str(e)}
        return err_json, None, (pd.DataFrame(history or [])), (history or [])

def export_history(history):
    """
    Returns a CSV file containing the history.
    """
    history = history or []
    if not history:
        # Create an empty CSV to keep UX consistent
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        pd.DataFrame([]).to_csv(tmp.name, index=False)
        return tmp.name

    df = pd.DataFrame(history)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

def clear_history():
    """
    Clears the history: returns an empty DF and an empty list state.
    """
    return pd.DataFrame([]), []

# -----------------------------
# 4️⃣ UI – Blocks with state + history table
# -----------------------------
with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        """
        # 💳 Bank Transaction Fraud Detector
        🔍 Predict whether a transaction is **Fraudulent** or **Normal** using a trained ML model.  
        Enter **30 features** below or try the ready-made examples.  
        Each prediction is saved in the **History** table with a timestamp and probabilities.
        """
    )

    # App state: a Python list of dicts we’ll keep appending to
    history_state = gr.State([])

    with gr.Row():
        with gr.Column():
            # 30 inputs
            inputs = [gr.Number(label=f"Feature {i+1}", value=0.0) for i in range(30)]

            # Examples (fill first 30 fields)
            gr.Examples(
                examples=[
                    [0.0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388,
                     0.239599, 0.098698, 0.363787, 0.090794, -0.551599, -0.617800, -0.991390,
                     -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 0.403993, -0.251412,
                     0.004102, -0.045549, 0.145356, -0.069083, 0.002305, 0.111813, 149.62, 0.0, 0.0],

                    [0.0, 1.191857, 0.266151, 0.166480, 0.448154, 0.060018, -0.082361,
                     -0.078803, 0.085102, -0.255425, -0.166974, 1.612727, 1.065235, 0.489095,
                     -0.143772, 0.635558, 0.463917, -0.114805, -0.183361, -0.145783, 0.063978,
                     0.056119, 0.068507, 0.123300, 0.037108, 0.042082, 2.69, 0.0, 0.0, 0.0]
                ],
                inputs=inputs,
                label="Try Example Transactions"
            )

            predict_btn = gr.Button("🔍 Predict", variant="primary")

        with gr.Column():
            result_json = gr.JSON(label="Prediction Result")
            prob_plot = gr.Plot(label="Fraud Probability Chart")

    gr.Markdown("## 📜 Prediction History")
    with gr.Row():
        history_df = gr.Dataframe(
            headers=["Timestamp", "Prediction", "Prob_Normal", "Prob_Fraud", "Features_Preview"],
            row_count=(0, "dynamic"),
            wrap=True,
            interactive=False
        )

    with gr.Row():
        download_btn = gr.Button("⬇️ Download History (CSV)")
        clear_btn = gr.Button("🧹 Clear History")
        file_out = gr.File(label="History CSV", visible=False)

    # Wiring
    predict_btn.click(
        fn=predict_and_log,
        inputs=inputs + [history_state],
        outputs=[result_json, prob_plot, history_df, history_state]
    )

    download_btn.click(
        fn=export_history,
        inputs=[history_state],
        outputs=[file_out]
    )

    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[history_df, history_state]
    )

# -----------------------------
# 5️⃣ Launch
# -----------------------------
if __name__ == "__main__":
    demo.launch(share=True)
