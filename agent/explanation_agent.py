from agent.verification_agent import is_displayable


def explanation_agent(state: dict) -> dict:
    supported = [
        check for check in state["checks"]
        if is_displayable(check)
    ]

    if not supported:
        return {"answer": "I couldn't find a supported answer in the retrieved passages."}

    unique_claims = {}

    for check in supported:
        claim = check["claim"].strip()
        key = " ".join(claim.casefold().split()).rstrip(".!?")
        item = unique_claims.setdefault(
            key, {"claim": claim, "evidence_ids": []}
        )
        item["evidence_ids"].extend(check["evidence_ids"])

    lines = []

    for item in unique_claims.values():
        lines.append(f"- {item['claim']}")
        references = {}

        for i in dict.fromkeys(item["evidence_ids"]):
            passage = state["evidence"][i - 1]
            location = (passage["source"], passage["page"])
            references.setdefault(location, []).append(i)

        for (source, page), ids in references.items():
            labels = ", ".join(f"[{i}]" for i in ids)
            lines.append(f"  Evidence reviewed {labels}: {source}, PDF page {page}")

    if len(supported) < len(state["checks"]):
        lines.append("Some draft claims were omitted because they were unsupported or did not answer the question.")

    return {"answer": "\n".join(lines)}
