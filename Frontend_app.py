import streamlit as st
import requests
import json

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="⚖️ Legal Document Risk Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# -------------------- HEADER --------------------
st.title("⚖️ Legal Document Risk Analyzer")
st.markdown("Upload a **legal PDF document** to get an automated summary and clause-level risk analysis.")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("📄 Upload your legal PDF", type=["pdf"])

if uploaded_file is not None:
    st.info("⏳ Uploading and analyzing your document...")

    # Send file to FastAPI backend
    with st.spinner("Analyzing document with AI models..."):
        response = requests.post(
            "http://127.0.0.1:8000/analyze",
            files={"pdf_file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )

    # -------------------- RESPONSE HANDLING --------------------
    if response.status_code == 200:
        data = response.json()

        # --- Summary ---
        st.subheader("🧾 Document Summary")
        summary = data.get("summary", "")
        st.success(summary if summary else "No summary available.")

        # --- Risk Scores ---
        st.subheader("⚠️ Top Risk Factors")
        risks = data.get("top_risks", [])

        if risks:
            for label, score in risks[:5]:
                color = (
                    "🔴 High Risk" if score > 0.7 else
                    "🟠 Medium Risk" if score > 0.4 else
                    "🟢 Low Risk"
                )
                st.write(f"**{label.title()}** — {color}")
                st.progress(score)
        else:
            st.warning("No risk scores available.")

        # --- Clauses ---
        st.subheader("📜 Clause-Level Analysis")
        with st.expander("View clause details"):
            clauses = data.get("clauses", [])
            for i, clause_info in enumerate(clauses[:50], 1):
                clause_text = clause_info.get("clause", "")
                scores = clause_info.get("scores", {})
                st.markdown(f"**Clause {i}:** {clause_text[:300]}...")
                st.json(scores)

    else:
        st.error(f"Server returned error {response.status_code}: {response.text}")

st.markdown("---")
st.caption("💼 Developed by YourName | Final Year Project 2025")
