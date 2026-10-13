# Customer Support Triage & Auto-Reply Agent

A Streamlit application that processes uploaded support tickets with a local Qwen instruct model and a retrieval-augmented (RAG) company knowledge base.

## Architecture

`TicketLoaderAgent` reads CSV or XLSX ticket uploads. `ClassifierAgent` assigns a category and urgency and applies mandatory review triggers for security, safety, legal action, and repeat billing. `KnowledgeBaseAgent` splits uploaded TXT/PDF policies, creates embeddings and retrieves policy passages using normalized FAISS similarity. `ResponderAgent` drafts policy-grounded responses. `ValidatorAgent` checks the content and provides a reason for rejection; the supervisor can revise a response once. `ActionAgent` saves drafts in a local outbox or human-review queue and writes an activity log. `MemoryAgent` keeps recent ticket statuses for the current session. `SupervisorAgent` orchestrates the workflow.

## Installation

Use Python 3.10–3.12 and preferably a CUDA-enabled GPU with sufficient VRAM. The model `Qwen/Qwen2.5-3B-Instruct` and the embedding model download on first use. CPU inference may be very slow and requires adequate RAM.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Upload a relevant company policy or FAQ (`.txt` or text-based `.pdf`).
2. Upload tickets (`.csv` or `.xlsx`) containing `ticket_id`, `customer_name`, `customer_email`, `subject`, `message`.
3. Choose an individual ticket and click **Run agent pipeline**, or process multiple tickets using **Batch run**.
4. Inspect retrieved policy passages and similarity scores, classifier source, validation explanation, execution trace, drafted reply, and activity dashboard.
5. Any flagged ticket is stored in `review_queue/`, while other reply drafts are stored in `outbox/`. Logs are written to `logs/`.

