from agent.verification_agent import verification_agent

claim = "Users can leave the network during delivery."
unrelated = "The network uses a shared link."

state = {
    "evidence": [{"text": f"{claim} {unrelated}"}],
    "claims": [
        {"claim": claim, "quote": claim, "evidence_ids": [1]},
        {"claim": claim, "quote": unrelated, "evidence_ids": [1]},
    ],
}

first, second = verification_agent(state)["checks"]
print("Supporting quote:", first["verdict"])
print("Unrelated quote:", second["verdict"])
print("Unrelated quote reason:", second["reason"])
