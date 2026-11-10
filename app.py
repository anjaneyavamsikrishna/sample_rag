"""Enterprise Knowledge Assistant - run with:  streamlit run app.py"""
import streamlit as st

import app_logic as logic
import config
from retrieval import embeddings, reranker, vector_store

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading AI models (first start only)...")
def warm_up() -> bool:
    embeddings.get_model()      # load once now, so the first question is not slow
    reranker.get_model()
    return True


warm_up()
st.session_state.setdefault("messages", [])        # one conversation PER browser session


def notify(result: dict) -> None:
    level, text = logic.describe_upload(result)
    {"success": st.success, "info": st.info, "error": st.error}[level](text)


def show_details(msg: dict, debug: bool) -> None:
    for warning in msg.get("warnings", []):
        st.warning(warning)
    sources = msg.get("sources", [])
    if msg.get("abstained"):
        st.caption(f"No supporting evidence found (best match score {msg.get('best_score')}).")
    elif msg.get("mode") == "extractive":
        st.caption("Extractive mode: no LLM is configured, so this is the best matching passage.")
    if sources:
        cited = sum(s["cited"] for s in sources)
        with st.expander(f"Sources ({cited} cited, {len(sources)} retrieved)"):
            for s in sorted(sources, key=lambda s: not s["cited"]):
                st.markdown(f"**[{s['id']}] {logic.source_title(s)}** - {'✅ cited' if s['cited'] else 'retrieved'}"
                            + (f" · score {s['score']}" if debug else ""))
                st.text(s["text"])
    if debug and msg.get("role") == "assistant":
        st.caption(f"searched as: {msg.get('standalone_question')} · mode: {msg.get('mode')}"
                   + (f" · error: {msg['error_detail']}" if msg.get("error_detail") else ""))


# ------------------------------ sidebar ------------------------------
with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload files", type=logic.SUPPORTED_TYPES, accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not uploads):
        with st.spinner("Reading, chunking and indexing..."):
            for r in logic.process_uploads([(f.name, f.getvalue()) for f in uploads]):
                notify(r)
    if st.button("Load the sample Nimbus Tech files"):
        with st.spinner("Indexing sample files..."):
            for r in logic.load_sample_data():
                notify(r)

    docs = vector_store.list_documents()
    st.subheader(f"Indexed ({len(docs)})")
    if not docs:
        st.caption("Nothing indexed yet.")
    for d in docs:
        left, right = st.columns([5, 1])
        left.write(f"{d['source']}  \n:gray[{d['chunks']} chunks]")
        if right.button("🗑", key=f"delete_{d['doc_id']}", help=f"Delete {d['source']}"):
            vector_store.delete_document(d["doc_id"])
            st.rerun()

    st.header("⚙️ Settings")
    scope = st.selectbox("Search in", ["All documents"] + [d["source"] for d in docs])
    debug = st.toggle("Show retrieval details")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(logic.llm_status())
    st.caption(f"Relevance threshold: {config.MIN_RERANK_SCORE}")

# ------------------------------ main area ------------------------------
st.title("📚 Enterprise Knowledge Assistant")
st.caption("Ask questions about your documents. Every answer cites its sources, and the assistant says so when it cannot find one.")

if not docs:
    st.info("Start by uploading documents in the sidebar, or click **Load the sample Nimbus Tech files**.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_details(msg, debug)

if question := st.chat_input("Ask a question about your documents", disabled=not docs):
    history = list(st.session_state.messages)                      # earlier turns only
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            result = logic.ask(question, history, None if scope == "All documents" else scope)
        reply = {"role": "assistant", "content": result["answer"], **{
            k: result.get(k) for k in ("sources", "warnings", "abstained", "mode", "best_score",
                                       "standalone_question", "error_detail")}}
        st.markdown(reply["content"])
        show_details(reply, debug)
    st.session_state.messages.append(reply)
