# llm_feedback_analyzer.py
import json
import subprocess
import time
import csv

# Load your public feedback (scraped from X, Facebook, news comments)
FEEDBACK_FILE = "feedback.json"      # Must contain "id", "text", "lang"
RESULTS_FILE = "llm_feedback_results.csv"

def build_prompt(user_message):
    return f"""You are an IT assistant monitoring citizen feedback about a government digital service.
Given a message in any language, decide if it describes a technical problem (e.g., system down, error, slow, payment failed).
Answer ONLY "YES" or "NO".

Examples:
- "The system says I didn’t pay!" → YES
- "When will my passport arrive?" → NO
- "Beni otsake mi-password le see le, minyee mike mihe awo
mi passport akaunt le mli lolo. Mike laptop kroko po bor mdeŋ shi nor
ko nor ko tsakeee." → YES
- "Portal yeyea menyo nam o" → NO
- "Nhyehye no mma me kwan se menwie m’akatua wor ak-
wammisa krataa no ho. Mede intanet nkitahodi a eye den na ereye eyi
afi Nkran" → YES

Now classify:
- "{user_message}" →"""

def query_llm(prompt, model="qwen:0.5b", timeout=30):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        output = result.stdout.strip().upper()
        if "YES" in output:
            return "YES"
        elif "NO" in output:
            return "NO"
        else:
            return "UNCERTAIN"
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return "ERROR"

def main():
    # Load feedback
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        feedbacks = json.load(f)

    results = []
    print(f"🔍 Analyzing {len(feedbacks)} feedback messages with LLM...")

    for item in feedbacks:
        msg_id = item["id"]
        text = item["text"]
        lang = item["lang"]

        prompt = build_prompt(text)
        pred = query_llm(prompt)
        print(f"ID: {msg_id} | Lang: {lang} | Pred: {pred}")

        results.append({
            "id": msg_id,
            "text": text,
            "language": lang,
            "llm_prediction": pred
        })
        time.sleep(0.5)  # Be CPU-friendly

    # Save
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "language", "llm_prediction"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ LLM analysis complete → {RESULTS_FILE}")

if __name__ == "__main__":
    main()