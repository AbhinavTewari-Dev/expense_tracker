import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Credit Card Bill Analyzer", layout="wide")
st.title("💳 Credit Card Bill Analyzer")

st.markdown("""
Upload your **credit card statement** to detect:
- Interest & penalty charges  
- Unnecessary / avoidable expenses  
- Subscriptions & recurring payments  
- High-value transactions  
- Overpayment patterns  

This analysis uses an **autonomous AI agent** with **transaction-level reasoning**.
""")

# --------------------------------------------------
# AI AGENTS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def credit_card_transaction_agent(txn):
    """
    Agent that analyzes ONE transaction with amount awareness
    """
    prompt = f"""
You are a professional credit card optimization AI.

Transaction details:
Description: {txn['Description']}
Amount: ₹{txn['Amount']}

Tasks:
1. Classify this charge as:
   - Normal
   - Avoidable
   - Excessive
   - Penalty / Interest
2. Explain WHY in simple terms.
3. If avoidable or excessive, estimate YEARLY loss if this repeats.

Respond strictly in this format:

Type: <Normal / Avoidable / Excessive / Penalty>
Explanation: <clear explanation>
Estimated Yearly Impact: <₹ amount or NA>
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return res.choices[0].message.content.strip()


def credit_card_summary_agent(summary):
    prompt = f"""
You are an expert credit card optimization AI agent.

User credit card data:
{summary}

IMPORTANT:
- Do NOT focus only on the single largest transaction.
- Identify SYSTEMIC and RECURRING cost leaks such as:
  - Interest charges
  - Late fees
  - Penalties
  - Subscriptions
- One-time purchases (like travel) are secondary unless excessive.

Tasks:
1. Identify the TOP 2–3 cost leaks by long-term financial impact.
2. Quantify each leak using amounts (monthly/yearly).
3. Give clear, numeric recommendations to reduce each leak.

Respond in structured bullet points.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return res.choices[0].message.content.strip()

# --------------------------------------------------
# Upload
# --------------------------------------------------
file = st.file_uploader(
    "📂 Upload Credit Card Statement (CSV)",
    type=["csv"]
)

if file:
    df = pd.read_csv(file)
    df["Amount"] = df["Amount"].astype(float)

    st.subheader("🧾 All Credit Card Transactions")
    st.dataframe(df, use_container_width=True)

    # --------------------------------------------------
    # Basic Separation
    # --------------------------------------------------
    charges = df[df["Amount"] > 0]   # credit card spends
    refunds = df[df["Amount"] < 0]

    total_spend = charges["Amount"].sum()

    # --------------------------------------------------
    # Detect Suspicious Transactions
    # --------------------------------------------------
    suspicious = charges[
        charges["Description"].str.contains(
            "interest|fee|penalty|subscription|forex|cash advance|late|overlimit",
            case=False, na=False
        )
        |
        (charges["Amount"] > charges["Amount"].quantile(0.95))
    ]

    # --------------------------------------------------
    # Summary Metrics
    # --------------------------------------------------
    st.subheader("📊 Credit Card Summary")
    a, b, c = st.columns(3)

    a.metric("Total Spend", f"₹{int(total_spend):,}")
    b.metric("Refunds", f"₹{int(abs(refunds['Amount'].sum())):,}")
    c.metric("Suspicious Charges", len(suspicious))

    # --------------------------------------------------
    # Transaction-Level AI Analysis
    # --------------------------------------------------
    st.subheader("🔍 Transaction-Level AI Analysis")

    if suspicious.empty:
        st.success("No high-risk or suspicious transactions detected.")
    else:
        for _, row in suspicious.iterrows():
            with st.expander(f"{row['Description']} — ₹{int(row['Amount'])}"):
                analysis = credit_card_transaction_agent(row)
                st.markdown(analysis)

    # --------------------------------------------------
    # Autonomous Credit Card Agent (FULL VIEW)
    # --------------------------------------------------
    st.subheader("🤖 Autonomous Credit Card Optimization Agent")

    summary = {
        "total_spend": int(total_spend),
        "top_charges": charges.sort_values("Amount", ascending=False).head(5)[
            ["Description", "Amount"]
        ].to_dict(orient="records"),
        "interest_and_fees": charges[
            charges["Description"].str.contains(
                "interest|fee|penalty|late|overlimit",
                case=False, na=False
            )
        ]["Amount"].sum(),
        "subscriptions": charges[
            charges["Description"].str.contains(
                "subscription|netflix|spotify|prime|membership|apple|google",
                case=False, na=False
            )
        ][["Description","Amount"]].to_dict(orient="records")
    }

    if st.button("Run Credit Card Agent"):
        agent_output = credit_card_summary_agent(summary)
        st.success(agent_output)

# --------------------------------------------------
st.caption("⚠️ Educational demo only. Not financial advice.")