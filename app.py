import streamlit as st
import time

from agents.retriever_agent import retriever_agent
from agents.fraud_agent import fraud_agent
from agents.risk_agent import risk_agent
from agents.report_agent import report_agent

# PAGE CONFIG
st.set_page_config(
    page_title="Multi-Agent Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #4CAF50;
}

.agent-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1E1E1E;
    margin-bottom: 10px;
}

.success-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #0E5A2A;
    color: white;
}

.report-box {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown('<p class="big-title">🛡️ Multi-Agent Fraud Detection System</p>', unsafe_allow_html=True)

st.write("AI-powered fraud investigation using autonomous agents.")

# SIDEBAR
st.sidebar.title("🤖 Agent Workflow")

st.sidebar.success("1. Retriever Agent")
st.sidebar.success("2. Fraud Analysis Agent")
st.sidebar.success("3. Risk Assessment Agent")
st.sidebar.success("4. Report Generation Agent")

st.sidebar.markdown("---")

st.sidebar.info("""
### Tech Stack
- Streamlit
- Gemini / OpenRouter
- FAISS
- Sentence Transformers
- Multi-Agent AI
""")

# USER INPUT
query = st.text_area(
    "Enter Fraud Investigation Query",
    placeholder="Example: Analyze suspicious transaction behavior..."
)

# BUTTON
if st.button("🚀 Run Investigation"):

    with st.spinner("Agents are analyzing..."):

        # AGENT 1
        st.info("Retriever Agent Running...")
        context = retriever_agent(query)

        time.sleep(1)

        # AGENT 2
        st.info("Fraud Agent Running...")
        answer = fraud_agent(context, query)

        time.sleep(1)

        # AGENT 3
        st.info("Risk Agent Running...")
        risk = risk_agent(answer)

        time.sleep(1)

        # AGENT 4
        st.info("Report Agent Running...")
        final_report = report_agent(query, answer, risk)

    st.success("✅ Investigation Completed")

    # RISK SCORE
    st.subheader("📊 Risk Score")

    risk_level = "HIGH"

    if "LOW" in risk.upper():
        risk_level = "LOW"
        st.success("LOW RISK")

    elif "MEDIUM" in risk.upper():
        risk_level = "MEDIUM"
        st.warning("MEDIUM RISK")

    else:
        st.error("HIGH RISK")

    # REPORT
    st.subheader("📄 Final Investigation Report")

    st.markdown(
        f"""
        <div class="report-box">
        <pre>{final_report}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    # DOWNLOAD BUTTON
    st.download_button(
        label="📥 Download Report",
        data=final_report,
        file_name="fraud_report.txt",
        mime="text/plain"
    )