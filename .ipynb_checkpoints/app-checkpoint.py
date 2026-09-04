import streamlit as st
import sqlite3
import pandas as pd
from chat import CFAAssistant, save_feedback

st.set_page_config(page_title="CFA Level II Assistant", page_icon="📈", layout="wide")

@st.cache_resource
def load_assistant():
    return CFAAssistant()

assistant = load_assistant()

# UI Navigation Tabs in English
tab1, tab2 = st.tabs(["💬 Study Assistant", "📊 System Metrics Dashboard"])

with tab1:
    st.title("📈 CFA Level II Study Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metrics" in message:
                st.caption(message["metrics"])

    # Chat input
    if prompt := st.chat_input("Ask a CFA Level II question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking & Evaluating..."):
                result = assistant.run_chat(prompt)
                st.session_state.conversation_id = result["conversation_id"]

                st.markdown(result["response"])
                st.caption(result["metrics_text"])

        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["response"],
            "metrics": result["metrics_text"]
        })
        st.rerun()

    # Human feedback buttons in English
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        st.write("---")
        st.markdown("**Rate this answer (Human Feedback):**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("+1 (Thumbs Up)", key=f"up_{conversation_id}"):
                save_feedback(conversation_id, source="user", score=1)
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("-1 (Thumbs Down)", key=f"down_{conversation_id}"):
                save_feedback(conversation_id, source="user", score=-1)
                st.warning("Thanks for the feedback! We will review this.")

with tab2:
    st.title("📊 Operational, Cost & Feedback Master Dashboard")
    
    conn = sqlite3.connect('metrics.db')
    df_metrics = pd.read_sql_query("SELECT * FROM llm_metrics", conn)
    df_feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()

    total_q = len(df_metrics)
    total_cost = df_metrics['cost'].sum() if total_q > 0 else 0
    avg_latency = df_metrics['response_time'].mean() if total_q > 0 else 0
    total_tokens = df_metrics['total_tokens'].sum() if total_q > 0 else 0
    
    # Top KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", total_q)
    col2.metric("Total Cost", f"${total_cost:.4f}")
    col3.metric("Avg Latency", f"{avg_latency:.2f}s")
    col4.metric("Total Tokens Used", f"{total_tokens:,}")

    st.divider()

    # Charts section
    if not df_metrics.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("⏱️ Response Latency (Seconds)")
            st.line_chart(df_metrics.set_index("timestamp")[["response_time"]])
        with c2:
            st.subheader("💰 Cumulative Cost ($)")
            df_sorted = df_metrics.sort_values("timestamp")
            df_sorted["cumulative_cost"] = df_sorted["cost"].cumsum()
            st.line_chart(df_sorted.set_index("timestamp")[["cumulative_cost"]])

    st.divider()
    st.subheader("📝 Raw Operational Metrics Logs")
    st.dataframe(df_metrics, use_container_width=True)

    st.subheader("🗣️ Unified Feedback Logs (User vs. Judge)")
    st.dataframe(df_feedback, use_container_width=True)