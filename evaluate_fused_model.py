# evaluate_fused_model.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

LOGS_FILE = "logs.csv"
FEEDBACK_RESULTS = "llm_feedback_results.csv"
OUTPUT_FILE = "fused_evaluation_results.csv"

def main():
    # Check if required files exist
    import os
    if not os.path.exists(LOGS_FILE):
        print(f"❌ Error: {LOGS_FILE} not found!")
        print("   Run: python generate_logs.py")
        return

    if not os.path.exists(FEEDBACK_RESULTS):
        print(f"❌ Error: {FEEDBACK_RESULTS} not found!")
        print("   This file should contain LLM feedback analysis results.")
        print("   Columns needed: id, text, language, llm_prediction")
        return

    # Load data
    print(f"✓ Loading {LOGS_FILE}...")
    logs = pd.read_csv(LOGS_FILE)
    print(f"✓ Loading {FEEDBACK_RESULTS}...")
    feedback = pd.read_csv(FEEDBACK_RESULTS)
    print(f"  - Logs: {len(logs)} deployments")
    print(f"  - Feedback: {len(feedback)} messages")

    # Prepare log features (matching the actual columns in logs.csv)
    features = [
        "build_duration_sec",
        "error_rate",
        "code_churn_lines",
        "test_coverage_pct",
        "latency_p95_ms",
        "failed_test_cases",
        "time_since_last_deployment_hours"
    ]
    X = logs[features].fillna(0)  # Handle any missing values
    y = logs["failure_label"]

    # Train XGBoost (as in your SLR)
    print("\n✓ Training XGBoost model on deployment logs...")
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y, verbose=False)
    logs["log_prob"] = model.predict_proba(X)[:, 1]
    print(f"  Model trained on {len(features)} features")

    # Align with feedback (simulate 1:1 for feasibility study)
    N = min(len(logs), len(feedback))
    print(f"\n✓ Aligning logs and feedback (using first {N} records)...")
    logs_sub = logs.head(N).reset_index(drop=True)
    feedback_sub = feedback.head(N).reset_index(drop=True)

    # Convert LLM output to signal
    feedback_sub["feedback_signal"] = feedback_sub["llm_prediction"].apply(
        lambda x: 1 if x == "YES" else 0
    )

    # Fused risk score (weights from your proposal's emphasis on human signals)
    logs_sub["fused_risk"] = 0.6 * logs_sub["log_prob"] + 0.4 * feedback_sub["feedback_signal"]
    logs_sub["fused_pred"] = (logs_sub["fused_risk"] >= 0.5).astype(int)
    logs_sub["log_only_pred"] = (logs_sub["log_prob"] >= 0.5).astype(int)

    # Evaluate
    y_true = logs_sub["failure_label"]
    print("\n📊 Logs-Only Model:")
    print(classification_report(y_true, logs_sub["log_only_pred"], digits=3))

    print("\n📊 Fused Model (Logs + LLM Feedback):")
    print(classification_report(y_true, logs_sub["fused_pred"], digits=3))

    # Save full results
    output = pd.concat([logs_sub, feedback_sub[["text", "language", "llm_prediction"]]], axis=1)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Full results saved → {OUTPUT_FILE}")

    # Show qualitative examples for your paper
    failures = output[output["failure_label"] == 1].head(3)
    if len(failures) > 0:
        print("\n🔍 Qualitative Examples (for your paper):")
        for _, row in failures.iterrows():
            text_preview = row['text'][:60] if len(str(row['text'])) > 60 else row['text']
            print(f"- [{row['language']}] \"{text_preview}...\" → LLM: {row['llm_prediction']}")
            print(f"  Log Prob: {row['log_prob']:.2f} | Fused Risk: {row['fused_risk']:.2f}\n")
    else:
        print("\n⚠ No failure examples found in the aligned dataset.")

if __name__ == "__main__":
    main()