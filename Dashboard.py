import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Personal Finance Advisor", layout="wide")
st.title("💸 AI Personal Finance Advisor")

# --------------------------------------------------
# Constants
# --------------------------------------------------
CATEGORIES = [
    "Food","Travel","Shopping","Rent","Bills",
    "Entertainment","Investment","Income","Other"
]
FIXED_CATEGORIES = ["Rent"]

RISK_RETURN = {
    "Conservative (8%)": 0.08,
    "Balanced (12%)": 0.12,
    "Aggressive (15%)": 0.15
}

# --------------------------------------------------
# Finance Math
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
    if principal <= 0:
        return 0
    r = (annual_rate / 100) / 12
    n = years * 12
    return int(principal * r * pow(1+r, n) / (pow(1+r, n) - 1))

def classify_cashflow(category):
    if category == "Income":
        return "Income"
    if category in ["Investment"]:
        return "Investment"
    return "Expense"

# --------------------------------------------------
# AI FUNCTIONS
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
1. Identify the single biggest financial issue OR strength.
2. Explain why.
3. Give ONE clear, actionable recommendation.

Praise good behavior where applicable.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

def ai_insight_agent(state):
    prompt = f"""
You are a senior financial insight AI.

User snapshot:
{state}

Tasks:
1. Highlight financial strengths and weaknesses.
2. If investment_rate >= 25%, explicitly praise discipline.
3. Identify the biggest risk.
4. Give high-impact recommendations.

Executive summary style.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

def ai_category_insight(category, spent, budget):
    prompt = f"""
Category: {category}
Spent: ₹{spent}
Budget: ₹{budget}

Explain overspend/underspend with numbers.
Give one specific corrective action.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return res.choices[0].message.content.strip()

def ai_financial_hygiene_score(state):
    prompt = f"""
Evaluate financial hygiene (0–100).

User:
{state}

Explain score and give 2 fixes.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

def ai_goal_strategy(goal, finances):
    prompt = f"""
Goal:
{goal}

User finances:
{finances}

Advise realistically.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

def ai_goal_feasibility(goal, sip, emi, savings):
    prompt = f"""
Goal: {goal}
SIP: ₹{sip}
EMI: ₹{emi}
Savings: ₹{savings}

Give feasibility score (0–100) and fix.
"""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content.strip()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
st.subheader("🧾 Financial Profile")
income = st.number_input("Monthly Income (₹)", 85000, step=5000)
emi_existing = st.number_input("Existing EMI (₹)", 15000, step=1000)
risk = st.selectbox("Risk Profile", list(RISK_RETURN.keys()))
exp_return = RISK_RETURN[risk]
emergency = st.number_input("Emergency Fund (₹)", 75000, step=5000)

st.subheader("💰 Category Budgets")
budgets = {
    c: st.number_input(f"{c} Budget", 5000, step=500)
    for c in CATEGORIES if c not in FIXED_CATEGORIES and c != "Income"
}

file = st.file_uploader("Upload Bank Statement CSV", type=["csv"])

# --------------------------------------------------
# STATE
# --------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "goals" not in st.session_state:
    st.session_state.goals = []

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if file and st.button("🚀 Analyze Expenses"):
    df = pd.read_csv(file)
    progress = st.progress(0)
    status = st.empty()

    cats, reasons = [], []
    for i, d in enumerate(df["Description"], 1):
        status.text(f"Analyzing {i}/{len(df)} transactions")
        c, r = ai_categorize_with_reason(d)
        cats.append(c)
        reasons.append(r)
        progress.progress(i / len(df))

    df["Category"] = cats
    df["Reason"] = reasons
    df["CashflowType"] = df["Category"].apply(classify_cashflow)

    st.session_state.df = df
    st.session_state.analyzed = True

    status.empty()
    progress.empty()

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if st.session_state.analyzed:
    df = st.session_state.df

    expense_df = df[(df["Amount"] < 0) & (df["CashflowType"] == "Expense")]
    investment_df = df[(df["Amount"] < 0) & (df["CashflowType"] == "Investment")]

    total_expenses = abs(expense_df["Amount"].sum())
    total_investments = abs(investment_df["Amount"].sum())
    savings = income - total_expenses - total_investments - emi_existing
    investment_rate = (total_investments / income) * 100 if income else 0

    st.subheader("🧾 Transaction Explainability")
    st.dataframe(df, width="stretch")

    # -------------------------------
    # Wealth Snapshot
    # -------------------------------
    st.subheader("📌 Wealth Snapshot")
    a,b,c,d,e = st.columns(5)
    a.metric("Income", f"₹{income:,}")
    b.metric("Expenses", f"₹{int(total_expenses):,}")
    c.metric("Investments", f"₹{int(total_investments):,}")
    d.metric("Savings", f"₹{int(savings):,}")
    e.metric("Investment Rate", f"{investment_rate:.1f}%")

    # -------------------------------
    # Charts (RESTORED)
    # -------------------------------
    st.subheader("📊 Spending Breakdown")

    cat_spend = expense_df.groupby("Category")["Amount"].sum().abs()
    daily_spend = expense_df.groupby("Date")["Amount"].sum().abs()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.pie(cat_spend, labels=cat_spend.index, autopct="%1.1f%%")
        ax.set_title("Category-wise Expense Split")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        daily_spend.plot(kind="bar", ax=ax)
        ax.set_title("Daily Expense Trend")
        st.pyplot(fig)

    # -------------------------------
    # Category-wise Analysis (RESTORED)
    # -------------------------------
    st.subheader("📂 Category-wise Spend Analysis")
    for category, spent in cat_spend.items():
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
            st.warning("No budget set")

        with st.expander("🤖 AI Explanation"):
            st.markdown(ai_category_insight(category, int(spent), int(budget)))

        st.divider()

    # -------------------------------
    # AI AGENTS
    # -------------------------------
    st.subheader("🧠 Insight Agent")
    if st.button("Generate Insight"):
        st.success(ai_insight_agent({
            "income": income,
            "expenses": total_expenses,
            "investments": total_investments,
            "investment_rate": investment_rate,
            "goals": st.session_state.goals
        }))

    st.subheader("🧠 Financial Hygiene Score")
    if st.button("Evaluate Hygiene"):
        st.success(ai_financial_hygiene_score({
            "income": income,
            "expenses": total_expenses,
            "emi": emi_existing,
            "savings": savings
        }))

    st.subheader("🤖 Autonomous Agent")
    if st.button("Run Autonomous Agent"):
        st.success(autonomous_finance_agent({
            "income": income,
            "expenses": total_expenses,
            "investments": total_investments,
            "investment_rate": investment_rate,
            "goals": st.session_state.goals
        }))

    # -------------------------------
    # Goals
    # -------------------------------
    st.subheader("🎯 Goal Planning")
    g_name = st.text_input("Goal Name")
    g_amt = st.number_input("Goal Amount", step=50000)
    g_years = st.slider("Years", 1, 15, 3)
    g_loan = st.checkbox("Use Loan")

    g_lp = st.slider("Loan %", 0, 80, 50) if g_loan else 0
    g_lr = st.slider("Loan Interest %", 6.0, 15.0, 10.0) if g_loan else 0

    if st.button("Add Goal"):
        st.session_state.goals.append({
            "name": g_name,
            "amount": g_amt,
            "years": g_years,
            "loan_pct": g_lp,
            "loan_rate": g_lr
        })

    for g in st.session_state.goals:
        loan = g["amount"] * g["loan_pct"] / 100
        sip = sip_required(g["amount"] - loan, g["years"], exp_return)
        emi = calculate_emi(loan, g["loan_rate"], g["years"])

        st.markdown(f"### 🎯 {g['name']}")
        st.write(f"SIP: ₹{sip:,} | EMI: ₹{emi:,}")
        st.info(ai_goal_strategy(g, {"savings": savings}))
        with st.expander("Goal Feasibility"):
            st.markdown(ai_goal_feasibility(g, sip, emi, savings))

st.caption("⚠️ Educational demo only. Not financial advice.")