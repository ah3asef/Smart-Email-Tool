"""Minimal Streamlit UI for the Smart Email Tool API."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_BASE_URL = "http://localhost:8000"


def post_query(payload: dict) -> dict:
    request = Request(
        f"{API_BASE_URL}/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def post_fetch(payload: dict) -> dict:
    request = Request(
        f"{API_BASE_URL}/fetch-emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


st.set_page_config(page_title="Smart Email Tool", page_icon="✉️")
st.title("Smart Email Tool")
st.caption("Ask questions about your inbox, or draft a reply, using your indexed emails.")

with st.form("query_form"):
    query = st.text_area(
        "Question",
        placeholder="e.g. What emails did I receive about DevOps jobs?",
        height=100,
    )
    col1, col2 = st.columns(2)
    mode = col1.radio("Mode", ["question_answering", "email_drafting"])
    top_k = col2.slider("Top results", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Send", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching your emails and generating an answer..."):
            try:
                result = post_query(
                    {
                        "query": query.strip(),
                        "mode": mode,
                        "top_k": top_k,
                    }
                )
            except HTTPError as exc:
                detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
                st.error(f"Request failed ({exc.code}): {detail}")
            except URLError:
                st.error(
                    f"Could not reach the API at {API_BASE_URL}. "
                    "Start it with: uvicorn api:app --host 0.0.0.0 --port 8000"
                )
            else:
                st.subheader("Answer")
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander(f"Sources ({len(result['sources'])})"):
                        for source in result["sources"]:
                            st.markdown(f"- **{source['subject'] or '(no subject)'}** — email `{source['email_id']}`, chunk `{source['chunk_id']}`")
                else:
                    st.caption("No sources retrieved.")

st.sidebar.header("Gmail Sync")
st.sidebar.caption(
    "Fetch the latest messages from Gmail and index them for search. "
    "On first run, Google's OAuth consent page opens in your browser."
)
with st.sidebar.form("fetch_form"):
    fetch_count = st.number_input("Max emails", min_value=1, max_value=500, value=50, step=1)
    fetch_label = st.text_input("Gmail label", value="INBOX")
    fetch_submitted = st.form_submit_button("Fetch new emails", type="primary")

if fetch_submitted:
    with st.spinner("Fetching, cleaning and indexing emails... this can take a minute."):
        try:
            result = post_fetch(
                {
                    "max_emails": int(fetch_count),
                    "label": fetch_label.strip() or "INBOX",
                }
            )
        except HTTPError as exc:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
            st.sidebar.error(f"Fetch failed ({exc.code}): {detail}")
        except URLError:
            st.sidebar.error(
                f"Could not reach the API at {API_BASE_URL}. "
                "Start it with: uvicorn api:app --host 0.0.0.0 --port 8000"
            )
        else:
            st.sidebar.success(
                f"Fetched {result['fetched']} emails — {result['chunks_indexed']} chunks indexed."
            )