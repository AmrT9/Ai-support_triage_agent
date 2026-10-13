
import os
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from agents import (
    ModelAgent,
    EmbeddingAgent,
    KnowledgeBaseAgent,
    TicketLoaderAgent,
    MemoryAgent,
    ActionAgent,
    SupervisorAgent,
)


st.set_page_config(
    page_title="Support Triage & Auto-Reply Agent",
    page_icon="🎧",
    layout="wide",
)


@st.cache_resource(
    show_spinner="Loading local AI model..."
)
def get_llm():
    return ModelAgent()


@st.cache_resource(
    show_spinner="Loading embedding model..."
)
def get_embedder():
    return EmbeddingAgent()


if "memory" not in st.session_state:
    st.session_state.memory = MemoryAgent()

if "action" not in st.session_state:
    st.session_state.action = ActionAgent()

if "kb" not in st.session_state:
    st.session_state.kb = None


st.title("🎧 Customer Support Triage & Auto-Reply Agent")

st.caption(
    "Multi-Agent AI System | Classification → "
    "RAG → Response Generation → Validation → Actions"
)


with st.sidebar:
    st.header("1. Knowledge Base")

    kb_file = st.file_uploader(
        "Upload FAQ / Policy Document",
        type=["pdf", "txt"],
    )

    st.header("2. Customer Tickets")

    tickets_file = st.file_uploader(
        "Upload customer support tickets",
        type=["csv", "xlsx"],
    )

    st.header("3. Options")

    save_reply = st.toggle(
        "Save replies to outbox or review queue",
        value=True,
    )

    load_btn = st.button(
        "Load / Reload Knowledge Base",
        use_container_width=True,
    )


if kb_file is None or tickets_file is None:
    st.info(
        "Upload a policy document and customer ticket "
        "file in the sidebar to begin."
    )
    st.stop()


os.makedirs("uploads", exist_ok=True)

kb_path = os.path.join(
    "uploads",
    "policy" + Path(kb_file.name).suffix.lower()
)

with open(kb_path, "wb") as f:
    f.write(kb_file.getbuffer())


tickets_path = os.path.join(
    "uploads",
    "tickets" + Path(tickets_file.name).suffix.lower()
)

with open(tickets_path, "wb") as f:
    f.write(tickets_file.getbuffer())


llm = get_llm()
embedder = get_embedder()


kb_signature = hashlib.sha256(
    kb_file.getvalue()
).hexdigest()


if (
    load_btn
    or st.session_state.kb is None
    or st.session_state.get("kb_signature") != kb_signature
):
    kb = KnowledgeBaseAgent(embedder)

    try:
        n_chunks = kb.ingest(kb_path)

    except Exception as exc:
        st.error(f"Unable to load policy: {exc}")
        st.stop()

    st.session_state.kb = kb
    st.session_state.kb_signature = kb_signature

    st.sidebar.success(
        f"Knowledge base loaded: {n_chunks} chunks"
    )


supervisor = SupervisorAgent(
    llm,
    st.session_state.kb,
    st.session_state.memory,
    st.session_state.action,
)


loader = TicketLoaderAgent()

try:
    tickets_df = loader.load(tickets_path)

except Exception as exc:
    st.error(f"Unable to load tickets: {exc}")
    st.stop()


st.sidebar.success(
    f"Loaded {len(tickets_df)} customer tickets"
)


tab_single, tab_batch, tab_dashboard = st.tabs([
    "📨 Single Ticket",
    "⚙️ Batch Processing",
    "📊 Dashboard",
])


URGENCY_ICON = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🔴",
}


# -------------------------------------------
# SINGLE TICKET PROCESSING
# -------------------------------------------

with tab_single:
    st.subheader("Select or Create a Ticket")

    options = (
        ["-- Custom Ticket --"]
        + tickets_df["ticket_id"].astype(str).tolist()
    )

    choice = st.selectbox(
        "Select Ticket",
        options
    )

    if choice == "-- Custom Ticket --":
        ticket = {
            "ticket_id": "CUSTOM-1",
            "customer_name": st.text_input(
                "Customer Name",
                "Alex"
            ),
            "customer_email": st.text_input(
                "Customer Email",
                "alex@example.com"
            ),
            "subject": st.text_input(
                "Subject",
                "Refund Question"
            ),
            "message": st.text_area(
                "Message",
                "Hi, I was charged twice. Can you help?"
            ),
        }

    else:
        row = tickets_df[
            tickets_df["ticket_id"].astype(str) == choice
        ].iloc[0]

        ticket = row.to_dict()

        st.write(f"**Subject:** {ticket['subject']}")
        st.write(f"**Message:** {ticket['message']}")

    if st.button(
        "Run Agent Pipeline",
        type="primary"
    ):
        with st.spinner("AI agents are processing..."):
            trace = supervisor.handle_ticket(
                ticket,
                save_reply=save_reply
            )

        classification = trace["classification"]

        category = classification["category"]
        urgency = classification["urgency"]

        if trace["verdict"] != "OK":
            st.warning(
                f"{URGENCY_ICON.get(urgency, '⚪')} "
                f"{category} | {urgency} Urgency | "
                "Human Review Required"
            )

        else:
            st.success(
                f"{URGENCY_ICON.get(urgency, '⚪')} "
                f"{category} | {urgency} Urgency"
            )

        st.caption(
            f"Classifier: "
            f"{classification.get('source', 'unknown')}"
        )

        st.write(
            "**Classification Reason:**",
            classification.get("reason", "")
        )

        # RAG results

        with st.expander(
            "🔍 Retrieved Knowledge Base Context"
        ):
            if trace["context"]:
                for chunk in trace["context"]:
                    st.markdown(
                        f"**Chunk {chunk['chunk_id']}** "
                        f"— Similarity: {chunk['score']:.3f}"
                    )

                    st.write(chunk["text"])

            else:
                st.warning(
                    "No relevant policy information found."
                )

        # Reply generation

        st.subheader("✉️ Generated Customer Reply")

        st.info(trace["reply"])

        st.write(
            "**Final Verdict:**",
            trace["verdict"]
        )

        st.write(
            "**Validation Reason:**",
            trace["validation"]["reason"]
        )

        # Agent execution trace

        with st.expander("🧠 Agent Execution Trace"):
            st.json(trace["steps"])

        # Automated actions

        st.subheader("⚙️ Automated Action")

        st.json(trace["log"])


# -------------------------------------------
# BATCH PROCESSING
# -------------------------------------------

with tab_batch:
    st.subheader("Batch Ticket Processing")

    st.dataframe(
        tickets_df,
        use_container_width=True
    )

    st.write(
        f"Total tickets available: {len(tickets_df)}"
    )

    if st.button("Run Agent on ALL Tickets"):
        progress = st.progress(0.0)

        results = []

        total = len(tickets_df)

        for i, row in tickets_df.iterrows():
            trace = supervisor.handle_ticket(
                row.to_dict(),
                save_reply=save_reply
            )

            results.append({
                "ticket_id": row["ticket_id"],
                "category": (
                    trace["classification"]["category"]
                ),
                "urgency": (
                    trace["classification"]["urgency"]
                ),
                "status": trace["log"]["status"],
                "validation": (
                    trace["validation"]["reason"]
                ),
                "reply_preview": (
                    trace["reply"][:80] + "..."
                ),
            })

            progress.progress(
                (i + 1) / total
            )

        results_df = pd.DataFrame(results)

        st.success(
            "Batch processing completed successfully."
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.download_button(
            "Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name="support_results.csv",
            mime="text/csv",
        )


# -------------------------------------------
# DASHBOARD
# -------------------------------------------

with tab_dashboard:
    st.subheader("📊 Activity Dashboard")

    log_df = st.session_state.action.read_log()

    st.dataframe(
        log_df,
        use_container_width=True
    )

    if not log_df.empty:
        total_processed = len(log_df)

        high_urgency = len(
            log_df[
                log_df["urgency"] == "High"
            ]
        )

        review_needed = len(
            log_df[
                log_df["status"] == "needs_human_review"
            ]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Processed",
                total_processed
            )

        with col2:
            st.metric(
                "High Urgency",
                high_urgency
            )

        with col3:
            st.metric(
                "Human Review Required",
                review_needed
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ticket Categories")

            st.bar_chart(
                log_df["category"].value_counts()
            )

        with col2:
            st.subheader("Urgency Distribution")

            st.bar_chart(
                log_df["urgency"].value_counts()
            )

        high = log_df[
            log_df["urgency"] == "High"
        ]

        if not high.empty:
            st.warning(
                f"🚨 {len(high)} high-urgency "
                "ticket records found."
            )

            st.dataframe(
                high,
                use_container_width=True
            )

        pending = log_df[
            log_df["status"] == "needs_human_review"
        ]

        if not pending.empty:
            st.subheader("👤 Human Review Queue")

            st.dataframe(
                pending,
                use_container_width=True
            )

    st.divider()

    st.subheader("🧠 Agent Memory")

    st.json(
        st.session_state.memory.as_list()
    )
