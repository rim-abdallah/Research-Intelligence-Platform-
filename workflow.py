from langgraph.graph import END, START, StateGraph

from agent.analysis_agent import analysis_agent
from agent.explanation_agent import explanation_agent
from agent.research_agent import ResearchState, research_agent
from agent.verification_agent import verification_agent
from argparse import ArgumentParser

class WorkflowState(ResearchState, total=False):
    claims: list[dict]
    checks: list[dict]
    answer: str
    comparison_field: str
    revision_attempted: bool
    


def after_verification(state: WorkflowState) -> str:
    # Stop if we already retried or have no passages to work with.
    if state.get("revision_attempted") or not state["evidence"]:
        return "explanation"

    # If at least one claim passed, show the supported claims.
    if any(check["verdict"] == "supported" for check in state["checks"]):
        return "explanation"

    # No claim passed, but evidence is available. Try one revision.
    return "analysis"


graph = StateGraph(WorkflowState)
graph.add_node("research", research_agent)
graph.add_node("analysis", analysis_agent)
graph.add_node("verification", verification_agent)
graph.add_node("explanation", explanation_agent)

graph.add_edge(START, "research")
graph.add_edge("research", "analysis")
graph.add_edge("analysis", "verification")
graph.add_conditional_edges(
    "verification",
    after_verification,
    {"analysis": "analysis", "explanation": "explanation"},
)
graph.add_edge("explanation", END)

workflow = graph.compile()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pdf", action="append", help="PDF path; repeat for more papers")
    args = parser.parse_args()

    question = input("Ask about the paper: ").strip()
    if not question:
        raise SystemExit("Enter a question first.")

    initial_state = {"question": question}
    if args.pdf:
        initial_state["pdf_paths"] = args.pdf

    result = workflow.invoke(initial_state)

    for i, item in enumerate(result["evidence"], start=1):
        text = item["text"].lower()
        if i == 5 or ("join" in text and "leav" in text):
            print(f"\nEvidence {i} | {item['source']} | PDF page {item['page']}")
            print(item["text"])

    for draft in result["claims"]:
        if "join" in draft["claim"].lower():
            print("\nSelected quote for the joining/leaving claim:")
            print(draft.get("quote", "No quote recorded."))


    if not result["checks"]:
        print("No draft claims to check.")

    for check in result["checks"]:
        print(f"\nDraft claim: {check['claim']}")
        print(f"Verdict: {check['verdict']}")
        print(f"Reason: {check['reason']}")
        print(f"Evidence IDs: {check['evidence_ids']}")

    print("Revision attempted:", result.get("revision_attempted", False))
    print("\nFinal answer:")
    print(result["answer"])
