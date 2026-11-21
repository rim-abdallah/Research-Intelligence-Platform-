def explanation_agent(state: dict) -> dict:
    supported = [
        check for check in state["checks"]
        if check["verdict"] == "supported" and check["evidence_ids"]
    ]

    if not supported:
        return {"answer": "No draft claim passed verification."}

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
        lines.append("Some draft claims did not pass verification and were omitted.")

    return {"answer": "\n".join(lines)}
