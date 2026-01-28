import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os, math
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Personal Finance Advisor", layout="wide")
st.title("💸 AI Personal Finance Advisor")
# st.sidebar.page_link(
#     "pages/credit_card_analyzer.py",
#     label="💳 Analyze Credit Card Bill"
# )

# --------------------------------------------------
# Constants
# --------------------------------------------------
CATEGORIES = ["Food","Travel","Shopping","Rent","Bills","Entertainment","Income","Other"]
FIXED_CATEGORIES = ["Rent"]

RISK_RETURN = {
    "Conservative (8%)": 0.08,
    "Balanced (12%)": 0.12,
    "Aggressive (15%)": 0.15
}

# --------------------------------------------------
# Finance Math (Deterministic)
# --------------------------------------------------
def sip_future_value(sip, years, rate):
    r = rate / 12
    n = years * 12
    return int(sip * ((pow(1+r, n) - 1) / r) * (1+r))

def sip_required(target, years, rate):
    r = rate / 12
    n = years * 12
    factor = ((pow(1+r, n) - 1) / r) * (1+r)
    return int(target / factor)

def calculate_emi(principal, annual_rate, years):
    r = (annual_rate / 100) / 12
    n = years * 12
    return int(principal * r * pow(1+r, n) / (pow(1+r, n) - 1))

# --------------------------------------------------
# AI Functions
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def ai_categorize_with_reason(desc):
    prompt = f"""
Categorize this transaction into one category from {CATEGORIES}.
Give a short reason.

Transaction: "{desc}"

Format:
Category: <category>
Reason: <reason>
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    text = res.choices[0].message.content
    try:
        cat = text.split("Category:")[1].split("\n")[0].strip()
        reason = text.split("Reason:")[1].strip()
    except:
        cat, reason = "Other", "Unable to classify"
    return cat if cat in CATEGORIES else "Other", reason

def autonomous_finance_agent(state):
    prompt = f"""
You are an autonomous personal finance advisor AI agent.

User financial state:
{state}

Tasks:
1. Identify the single biggest financial issue.
2. Explain why it is happening.
3. Give ONE clear, actionable recommendation.

Be professional, realistic, and concise.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

def ai_goal_strategy(goal, finances):
    prompt = f"""
You are a financial planning AI.

Goal details:
{goal}

User finances:
{finances}

Decide:
- Is the goal realistic?
- Should loan or investment dominate?
- What adjustment would help?

Give clear advice.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

# --------------------------------------------------
# 🔹 ADDED AI FUNCTION (CATEGORY ANALYSIS ONLY)
# --------------------------------------------------
def ai_category_insight(category, spent, budget):
    prompt = f"""
You are a professional personal finance advisor.

Category: {category}
Actual Monthly Spend: ₹{spent}
Budgeted Amount: ₹{budget}

Rules:
- Do NOT give generic advice.
- Use NUMBERS.
- Mention the exact overspend or underspend.
- Suggest a concrete behavioral change specific to this category.

Tasks:
1. State whether the user is over or under budget (with ₹ amount).
2. Explain the most likely reason.
3. Give ONE specific, measurable action (e.g., reduce X by Y%).

Respond in 3 short bullet points.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return res.choices[0].message.content.strip()

# --------------------------------------------------
# Inputs
# --------------------------------------------------
st.subheader("🧾 Financial Profile")
income = st.number_input("Monthly Income (₹)", 85000, step=5000)
emi_existing = st.number_input("Existing Monthly EMI (₹)", 15000, step=1000)
risk = st.selectbox("Risk Profile", list(RISK_RETURN.keys()))
exp_return = RISK_RETURN[risk]
emergency = st.number_input("Emergency Fund (₹)", 75000, step=5000)

st.subheader("💰 Category Budgets")
budgets = {}
for c in CATEGORIES:
    if c not in FIXED_CATEGORIES and c != "Income":
        budgets[c] = st.number_input(f"{c} Budget", 5000, step=500)

file = st.file_uploader("Upload Bank Statement CSV", type=["csv"])

# --------------------------------------------------
# State
# --------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "goals" not in st.session_state:
    st.session_state.goals = []

# --------------------------------------------------
# Analysis
# --------------------------------------------------
if file and st.button("🚀 Analyze Expenses"):
    df = pd.read_csv(file)
    progress = st.progress(0)
    status = st.empty()

    cats, reasons = [], []
    total = len(df)

    for i, d in enumerate(df["Description"], start=1):
        status.text(f"Analyzing transaction {i}/{total}")
        c, r = ai_categorize_with_reason(d)
        cats.append(c)
        reasons.append(r)
        progress.progress(i/total)

    df["Category"] = cats
    df["Reason"] = reasons

    exp_df = df[df["Amount"] < 0]
    cat_spend = exp_df.groupby("Category")["Amount"].sum().abs()

    st.session_state.df = df
    st.session_state.exp_df = exp_df
    st.session_state.cat_spend = cat_spend
    st.session_state.analyzed = True

    status.empty()
    progress.empty()

# --------------------------------------------------
# Results
# --------------------------------------------------
if st.session_state.analyzed:
    df = st.session_state.df
    exp_df = st.session_state.exp_df
    cat_spend = st.session_state.cat_spend

    total_spend = abs(exp_df["Amount"].sum())
    savings = income - total_spend - emi_existing

    # Transaction explainability
    st.subheader("🧾 Transaction-Level Explainability")
    st.dataframe(df[["Date","Description","Amount","Category","Reason"]], width="stretch")

    # Wealth
    sip_proj = sip_future_value(10000, 10, exp_return)
    st.subheader("📌 Wealth Snapshot")
    a,b,c,d = st.columns(4)
    a.metric("Income", f"₹{income:,}")
    b.metric("Expenses", f"₹{int(total_spend):,}")
    c.metric("Current Wealth", f"₹{emergency:,}")
    d.metric("Projected Wealth", f"₹{int(emergency+sip_proj):,}")

    # Charts
    st.subheader("📊 Spending Breakdown")
    x,y = st.columns(2)
    with x:
        fig,ax = plt.subplots()
        ax.pie(cat_spend, labels=cat_spend.index, autopct="%1.1f%%")
        st.pyplot(fig)
    with y:
        daily = exp_df.groupby("Date")["Amount"].sum().abs()
        fig,ax = plt.subplots()
        daily.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # --------------------------------------------------
    # 🔹 ADDED CATEGORY-WISE ANALYSIS SECTION
    # --------------------------------------------------
    st.subheader("📂 Category-Wise Spend Analysis")

    for category, spent in cat_spend.items():

        if category == "Income":
            continue

        budget = budgets.get(category, 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Category", category)
        c2.metric("Spent", f"₹{int(spent):,}")
        c3.metric("Budget", f"₹{int(budget):,}")

        if budget > 0:
            if spent > budget:
                st.error(f"Over budget by ₹{int(spent - budget):,}")
            else:
                st.success(f"Under budget by ₹{int(budget - spent):,}")
        else:
            st.warning("No budget set for this category")

        with st.expander("🤖 AI Explanation & Recommendation"):
            st.markdown(
                ai_category_insight(
                    category,
                    int(spent),
                    int(budget)
                )
            )

        st.divider()

    # Autonomous Agent
    st.subheader("🤖 Autonomous Financial Insight Agent")
    state = {
        "income": income,
        "expenses": int(total_spend),
        "savings_capacity": int(savings),
        "category_spend": cat_spend.to_dict(),
        "goals": st.session_state.goals
    }
    if st.button("Run Agent"):
        agent_output = autonomous_finance_agent(state)
        st.success(agent_output)

    # Goals
    st.subheader("🎯 Goal-Based Planning")
    st.markdown("### 🎯 Add New Goal")
    g_name = st.text_input("Goal Name")
    g_amt = st.number_input("Goal Amount", step=50000)
    g_years = st.slider("Time Horizon (Years)", 1, 15, 3)

    g_loan = st.checkbox("Use Loan")

    if g_loan:
        g_loan_pct = st.slider("Loan %", 0, 80, 50)
        g_loan_rate = st.slider("Loan Interest %", 6.0, 15.0, 10.0)
    else:
        g_loan_pct = 0
        g_loan_rate = 0.0

    if st.button("Add Goal"):
        st.session_state.goals.append({
            "name": g_name,
            "amount": g_amt,
            "years": g_years,
            "loan_pct": g_loan_pct,
            "loan_rate": g_loan_rate
        })


    for g in st.session_state.goals:
        loan_amt = g["amount"] * g["loan_pct"] / 100
        invest_amt = g["amount"] - loan_amt
        sip_req = sip_required(invest_amt, g["years"], exp_return)
        emi_goal = calculate_emi(loan_amt, g["loan_rate"], g["years"]) if loan_amt > 0 else 0

        st.markdown(f"""
### 🎯 {g['name']}
- Required SIP: ₹{sip_req:,}
- Goal EMI: ₹{emi_goal:,}
""")

        ai_advice = ai_goal_strategy(g, {"savings_capacity": savings})
        st.info(ai_advice)

# --------------------------------------------------
st.caption("⚠️ Educational demo only. Not financial advice.")