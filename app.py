import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="AI Expense Categorizer", layout="wide")
st.title("💸 AI Expense Categorizer & Spending Insights")

# ---------------------------
# Categories
# ---------------------------
CATEGORIES = [
    "Food",
    "Travel",
    "Shopping",
    "Rent",
    "Bills",
    "Entertainment",
    "Income",
    "Other"
]

# ---------------------------
# AI Functions
# ---------------------------
@st.cache_data(show_spinner=False)
def categorize_expense(description: str) -> str:
    prompt = f"""
    Categorize the expense into EXACTLY ONE category
    from this list:
    {', '.join(CATEGORIES)}

    Expense description: "{description}"

    Return ONLY the category name.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


@st.cache_data(show_spinner=False)
def generate_insights(df: pd.DataFrame) -> str:
    data = df[["Description", "Amount", "Category"]].to_string(index=False)

    prompt = f"""
    You are a personal finance assistant.

    Based on the expenses below, provide:
    1. Biggest spending category
    2. One overspending pattern
    3. Two practical money-saving tips

    Expenses:
    {data}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "📤 Upload your expense CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df, width="stretch")

    if st.button("🤖 Categorize Expenses"):
        with st.spinner("Analyzing expenses using OpenAI..."):
            df["Category"] = df["Description"].apply(categorize_expense)

        st.success("✅ Expenses categorized successfully!")

        # ---------------------------
        # Categorized Data
        # ---------------------------
        st.subheader("🧾 Categorized Expenses")
        st.dataframe(df, width="stretch")

        # ---------------------------
        # Charts
        # ---------------------------
        st.subheader("📊 Spending Analysis")

        col1, col2 = st.columns(2)

        with col1:
            expense_df = df[df["Amount"] < 0]
            category_spend = expense_df.groupby("Category")["Amount"].sum().abs()

            fig1, ax1 = plt.subplots()
            ax1.pie(
                category_spend,
                labels=category_spend.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax1.axis("equal")
            st.pyplot(fig1)

        with col2:
            daily_spend = df.groupby("Date")["Amount"].sum()

            fig2, ax2 = plt.subplots()
            daily_spend.plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Amount")
            ax2.set_xlabel("Date")
            st.pyplot(fig2)

        # ---------------------------
        # AI Insights
        # ---------------------------
        st.subheader("🧠 AI Spending Insights")
        insights = generate_insights(df)
        st.write(insights)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("⚠️ Informational purposes only. Not financial advice.")