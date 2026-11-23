from agent.analysis_agent import analysis_agent
from agent.research_agent import research_agent
from agent.verification_agent import verification_agent

state = {
    "question": "How does LightGCN combine embeddings from different layers?"
}
state.update(research_agent(state))

passage = next(
    (
        item for item in state["evidence"]
        if "weighted sum of its embeddings on all layers" in item["text"].lower()
    ),
    None,
)

if passage is None:
    raise SystemExit("The explicit weighted-sum passage was not retrieved.")

state["evidence"] = [passage]
print(f"Using only the passage from PDF page {passage['page']}")

state.update(analysis_agent(state))
print("Draft claims:", state["claims"])

state.update(verification_agent(state))
print("Checks:", state["checks"])
