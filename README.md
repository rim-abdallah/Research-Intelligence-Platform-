# Research Intelligence Platform

A local research assistant for asking questions about scientific papers and checking the evidence behind an answer.

## What works

- Upload several PDFs and choose which papers to search.
- Ask questions using a four-stage LangGraph workflow.
- Open a cited PDF page and highlight the selected quote when it can be located.
- Run an experimental terminal comparison that extracts each paper's problem and method separately.

## Run locally on Windows

Use Python 3.12. Install and start [Ollama](https://ollama.com/download), then run these commands in PowerShell from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed in PowerShell. Upload a PDF with selectable text, choose it under **Search in**, and ask a question. Use **Open cited evidence** to inspect the source page. The first run may also download the embedding and reranking models.

## How it works

1. **Research** retrieves candidate chunks and reranks shorter snippets.
2. **Analysis** drafts claims and selects a snippet for each claim.
3. **Verification** checks each claim against the selected quote, not the entire surrounding passage.
4. **Explanation** displays claims marked as supported and groups duplicate claims and references.

If no claim passes and retrieved evidence is available, the workflow allows one revision before withholding the answer.

## Limitations

The verifier is model-based and can make mistakes. A highlight shows where the quote appears, not that the claim is correct. Scanned PDFs without selectable text need OCR. Comparison is experimental and can miss or omit fields. The browser shows a highlighted page preview, not a full PDF viewer.

Keep private papers, credentials, and generated evidence previews out of Git.
