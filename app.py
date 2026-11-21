from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory

import streamlit as st
from pdf_evidence import render_evidence_page

from workflow import workflow


st.title("Research Intelligence Platform")

papers = st.file_uploader(
    "Upload PDF papers",
    type=["pdf"],
    accept_multiple_files=True,
)

available = [paper.name for paper in papers]
selected = st.multiselect("Search in", available, default=available)

with st.form("research_question"):
    question = st.text_input("Ask a question", max_chars=2000)
    submitted = st.form_submit_button("Search uploaded papers")

if submitted:
    papers = [paper for paper in papers if paper.name in selected]

    if not papers or not question.strip():
        st.warning("Select at least one PDF and enter a question.")
        st.stop()

    if len(papers) > 5 or any(paper.size > 25 * 1024 * 1024 for paper in papers):
        st.warning("Use up to five PDFs, at most 25 MiB each.")
        st.stop()

    names = [PureWindowsPath(paper.name).name for paper in papers]
    if len({name.casefold() for name in names}) != len(names):
        st.warning("Rename PDFs that have the same filename before uploading them.")
        st.stop()

    with TemporaryDirectory(prefix="research-pdfs-") as folder:
        pdf_paths = []

        for paper, name in zip(papers, names):
            data = paper.getvalue()
            if (
                not name.lower().endswith(".pdf")
                or PureWindowsPath(name).is_reserved()
                or any(char in name for char in ':*?"<>|')
                or not data[:1024].lstrip().startswith(b"%PDF-")
            ):
                st.error("One of the uploads is not a usable PDF.")
                st.stop()

            path = Path(folder) / name
            path.write_bytes(data)
            pdf_paths.append(str(path))

        with st.spinner("Searching and checking the evidence..."):
            result = workflow.invoke({
                "question": question.strip(),
                "pdf_paths": pdf_paths,
            })

    st.session_state["last_run"] = (
        result,
        {name: paper.getvalue() for name, paper in zip(names, papers)},
    )
    st.session_state.pop("citation_choice", None)

if "last_run" in st.session_state:
    result, pdf_bytes_by_name = st.session_state["last_run"]

    st.subheader("Answer")
    st.text(result["answer"])

    citations = {}

    for draft, check in zip(result["claims"], result["checks"]):
        if check["verdict"] != "supported" or not draft.get("quote"):
            continue

        quote = draft["quote"]

        for ref in check["evidence_ids"]:
            passage = result["evidence"][ref - 1]
            source, page = passage["source"], passage["page"]
            key = (ref, source, page, quote)
            citations[key] = f"[{ref}] {source}, page {page}: {quote[:70]}"

    if citations:
        choice = st.selectbox(
            "Open cited evidence",
            list(citations),
            format_func=citations.__getitem__,
            index=None,
            key="citation_choice",
        )

        if choice is not None:
            _, source, page, quote = choice
            png, found = render_evidence_page(
                pdf_bytes_by_name[source], page, quote
            )

            if not found:
                st.warning(
                    "The quote could not be located on this page. "
                    "No highlight was added."
                )

            st.image(png, caption=f"{source}, PDF page {page}")
            st.text(quote)
