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
st.title("💸 AI Expense Analyzer & Personal Finance Advisor")

st.markdown(
    """
    A smart personal finance system that categorizes expenses,
    monitors budgets, and provides professional-grade financial recommendations.
    """
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
CATEGORIES = [
    "Food", "Travel", "Shopping", "Rent",
    "Bills", "Entertainment", "Income", "Other"
]

FIXED_CATEGORIES = ["Rent"]

# --------------------------------------------------
# Rule-Based Categorization
# --------------------------------------------------
def rule_based_categorize(description: str) -> str:
    d = description.lower()

    if any(x in d for x in ["zomato", "swiggy", "restaurant", "grocery"]):
        return "Food"
    if any(x in d for x in ["uber", "ola", "rapido", "train", "flight"]):
        return "Travel"
    if any(x in d for x in ["amazon", "flipkart", "myntra", "store"]):
        return "Shopping"
    if "rent" in d:
        return "Rent"
    if any(x in d for x in ["electricity", "water", "recharge", "bill"]):
        return "Bills"
    if any(x in d for x in ["netflix", "spotify", "movie"]):
        return "Entertainment"
    if any(x in d for x in ["salary", "credit"]):
        return "Income"

    return "Other"

# --------------------------------------------------
# AI Categorization (Explainable)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def ai_categorize_with_reason(description: str):
    prompt = f"""
Categorize the expense below into ONE category only
from the list:
{', '.join(CATEGORIES)}

Expense: "{description}"

Respond exactly in this format:
Category: <category>
Reason: <short reason>
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    try:
        category = text.split("Category:")[1].split("\n")[0].strip()
        reason = text.split("Reason:")[1].strip()
    except:
        category, reason = "Other", "Unable to classify confidently."

    if category not in CATEGORIES:
        category = "Other"

    return category, reason

# --------------------------------------------------
# PROFESSIONAL AUTONOMOUS FINANCIAL AGENT
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def financial_advisor_agent(
    df: pd.DataFrame,
    category_budgets: dict,
    income: int,
    emi: int,
    risk_profile: str
):
    expense_df = df[df["Amount"] < 0]
    total_expense = abs(expense_df["Amount"].sum())

    savings_capacity = income - total_expense - emi

    discretionary_df = expense_df[
        ~expense_df["Category"].isin(FIXED_CATEGORIES)
    ]

    overspend_details = []
    for cat, spent in (
        discretionary_df.groupby("Category")["Amount"].sum().abs().items()
    ):
        budget = category_budgets.get(cat, 0)
        if budget > 0 and spent > budget:
            overspend_details.append(
                f"{cat}: ₹{int(spent)} vs budget ₹{int(budget)}"
            )

    prompt = f"""
You are a professional personal financial advisor.

User profile:
- Monthly income: ₹{income}
- Monthly EMIs: ₹{emi}
- Risk profile: {risk_profile}

Financial summary:
- Total monthly expenses: ₹{int(total_expense)}
- Estimated savings capacity: ₹{int(savings_capacity)}

Overspending (discretionary categories):
{chr(10).join(overspend_details) if overspend_details else "None"}

Tasks:
1. Decide whether user should prioritize expense reduction, debt reduction, or investments
2. Suggest a realistic SIP amount (if applicable)
3. Advise on EMI or loan prepayment (if applicable)
4. Give ONE immediate next step

Rules:
- Be numeric and practical
- Avoid generic advice
- Do NOT suggest cutting rent
- Keep response under 6 bullet points
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

# --------------------------------------------------
# CATEGORY EXPLANATION (PRO)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def explain_category_spend(category, amount, budget):
    prompt = f"""
A user spent ₹{int(amount)} on {category} this month.
Their budget for this category is ₹{int(budget)}.

Explain in ONE sentence why spending is high,
then suggest ONE realistic way to reduce it.

Avoid generic tips.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
use_ai = st.checkbox("Use AI-based categorization", value=True)

st.subheader("🧾 Financial Profile")

monthly_income = st.number_input(
    "Monthly Income (₹)", value=85000, step=5000
)

monthly_emi = st.number_input(
    "Total EMIs / Loans per month (₹)", value=15000, step=1000
)

risk_profile = st.selectbox(
    "Investment Preference",
    ["Conservative", "Balanced", "Aggressive"]
)

st.subheader("💰 Category Budgets")

category_budgets = {}
for cat in CATEGORIES:
    if cat not in FIXED_CATEGORIES and cat != "Income":
        category_budgets[cat] = st.number_input(
            f"{cat} budget (₹)",
            value=10000 if cat == "Food" else 5000,
            step=500
        )

uploaded_file = st.file_uploader(
    "📂 Upload bank statement CSV", type=["csv"]
)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if uploaded_file and st.button("🚀 Analyze Expenses"):
    df = pd.read_csv(uploaded_file)

    categories, reasons = [], []

    with st.spinner("Analyzing transactions..."):
        for desc in df["Description"]:
            if use_ai:
                cat, reason = ai_categorize_with_reason(desc)
            else:
                cat = rule_based_categorize(desc)
                reason = "Rule-based categorization"

            categories.append(cat)
            reasons.append(reason)

    df["Category"] = categories
    df["Reason"] = reasons

    expense_df = df[df["Amount"] < 0]
    cat_spend = expense_df.groupby("Category")["Amount"].sum().abs()

    st.session_state.df = df
    st.session_state.expense_df = expense_df
    st.session_state.cat_spend = cat_spend
    st.session_state.analyzed = True

    st.success("Analysis complete!")

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if st.session_state.analyzed:

    df = st.session_state.df
    expense_df = st.session_state.expense_df
    cat_spend = st.session_state.cat_spend

    total_spend = abs(expense_df["Amount"].sum())

    col1, col2 = st.columns(2)
    col1.metric("Total Monthly Spend", f"₹{int(total_spend)}")
    col2.metric("Top Spending Category", cat_spend.idxmax())

    st.subheader("🧾 Categorized Transactions")
    st.dataframe(df, width="stretch")

    st.subheader("📊 Spending Breakdown")

    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots()
        ax1.pie(cat_spend, labels=cat_spend.index, autopct="%1.1f%%")
        ax1.axis("equal")
        st.pyplot(fig1)

    with colB:
        daily = df.groupby("Date")["Amount"].sum()
        fig2, ax2 = plt.subplots()
        daily.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    st.subheader("🤖 Autonomous Financial Advisor")
    st.info(
        financial_advisor_agent(
            df,
            category_budgets,
            monthly_income,
            monthly_emi,
            risk_profile
        )
    )

    st.subheader("🔍 Category Deep Dive")
    selected_category = st.selectbox(
        "Select a category", cat_spend.index
    )

    if selected_category in category_budgets:
        if st.button("Analyze Selected Category"):
            st.info(
                explain_category_spend(
                    selected_category,
                    cat_spend[selected_category],
                    category_budgets.get(selected_category, 0)
                )
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("⚠️ Educational demo only. Not financial advice.")