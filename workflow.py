from langgraph.graph import END, START, StateGraph

from agent.analysis_agent import analysis_agent
from agent.explanation_agent import explanation_agent
from agent.research_agent import ResearchState, research_agent
from agent.verification_agent import verification_agent


class WorkflowState(ResearchState, total=False):
    claims: list[dict]
    checks: list[dict]
    answer: str
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
    question = input("Ask about LightGCN: ").strip()
    if not question:
        raise SystemExit("Enter a question first.")

    result = workflow.invoke({"question": question})

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
