def explanation_agent(state: dict) -> dict:
    supported = [
        check for check in state["checks"]
        if check["verdict"] == "supported" and check["evidence_ids"]
    ]

    if not supported:
        return {"answer": "No draft claim passed verification."}

    lines = []
    for check in supported:
        lines.append(f"- {check['claim']}")
        for i in dict.fromkeys(check["evidence_ids"]):
            passage = state["evidence"][i - 1]
            lines.append(
                f"  Evidence reviewed [{i}]: "
                f"{passage['source']}, PDF page {passage['page']}"
            )

    if len(supported) < len(state["checks"]):
        lines.append("Some draft claims did not pass verification and were omitted.")

    return {"answer": "\n".join(lines)}
