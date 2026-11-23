import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.verification_agent import verification_agent


CLAIM = "Users can leave the network during delivery."
UNRELATED = "The network uses a shared link."


def make_state(quote):
    return {
        "evidence": [{"text": f"{CLAIM} {UNRELATED}"}],
        "claims": [
            {"claim": CLAIM, "quote": quote, "evidence_ids": [1]}
        ],
    }


class VerificationTests(unittest.TestCase):
    @patch("agent.verification_agent.chat")
    def test_only_the_selected_quote_is_sent(self, mock_chat):
        mock_chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({
                "verdict": "insufficient",
                "reason": "The quote does not address the claim.",
            }))
        )

        result = verification_agent(make_state(UNRELATED))

        mock_chat.assert_called_once()
        messages = mock_chat.call_args.kwargs["messages"]
        payload = json.loads(messages[-1]["content"])
        self.assertEqual(
            payload["cited_passages"],
            [{"id": 1, "text": UNRELATED}],
        )
        self.assertEqual(result["checks"][0]["verdict"], "insufficient")

    @patch("agent.verification_agent.chat")
    def test_missing_quote_is_rejected_without_ollama(self, mock_chat):
        result = verification_agent(
            make_state("This quote is not in the passage.")
        )

        mock_chat.assert_not_called()
        self.assertEqual(result["checks"][0]["verdict"], "insufficient")
        self.assertEqual(result["checks"][0]["evidence_ids"], [])

