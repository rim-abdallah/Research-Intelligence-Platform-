from argparse import ArgumentParser
from pathlib import Path

from workflow import workflow


QUESTIONS = {
    "Problem": (
        "What specific research problem does this paper address? "
        "State the problem, not the proposed method."
    ),
    "Method": (
        "What is the paper's main proposed method or algorithm? "
        "Distinguish it from any baseline used only for comparison."
    ),
}


def compare_papers(pdf_paths: list[str], overview_only: bool = False) -> list[dict]:
    rows = []

    for path in pdf_paths:
        fields = {}
        for label, question in QUESTIONS.items():
            fields[label] = workflow.invoke({
                "question": question,
                "pdf_paths": [path],
                "comparison_field": label,
                "overview_only": overview_only,
            })


        rows.append({"paper": Path(path).name, "fields": fields})

    return rows


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pdf", action="append", required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--overview-only", action="store_true")
    args = parser.parse_args()

    if len(args.pdf) != 2:
        parser.error("Supply --pdf twice, once for each paper.")

    for row in compare_papers(args.pdf, overview_only=args.overview_only):
        print(f"\nPaper: {row['paper']}")
        for label, result in row["fields"].items():
            print(f"\n{label}:")
            print(result["answer"])
            if args.trace and label == "Method":
                print("Ranked method snippets:")
                for rank, snippet in enumerate(result["snippets"], start=1):
                    ref = snippet["passage_id"]
                    page = result["evidence"][ref - 1]["page"]
                    print(f"{rank}. PDF page {page}, passage {ref}: {snippet['text']}")

