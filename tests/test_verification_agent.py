import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.explanation_agent import explanation_agent
from agent.verification_agent import is_displayable, verification_agent
from workflow import after_verification


CLAIM = "Users can leave the network during delivery."
UNRELATED = "The network uses a shared link."


def make_state(quote):
    return {
        "question": "Can users leave during delivery?",
        "evidence": [{"text": f"{CLAIM} {UNRELATED}", "source": "test.pdf", "page": 1}],
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
                "answers_question": True,
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

    @patch("agent.verification_agent.chat")
    def test_supported_but_off_topic_claim_triggers_retry_then_abstains(self, mock_chat):
        mock_chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({
                "verdict": "supported",
                "answers_question": False,
                "reason": "The claim does not identify a GPU model.",
            }))
        )
        question = "What GPU model was used to train LightGCN?"
        claim = "LightGCN is a designed model."
        state = {
            "question": question,
            "evidence": [{"text": claim, "source": "test.pdf", "page": 3}],
            "claims": [{"claim": claim, "quote": claim, "evidence_ids": [1]}],
        }
        state.update(verification_agent(state))
        payload = json.loads(mock_chat.call_args.kwargs["messages"][-1]["content"])
        self.assertEqual(payload["question"], question)
        check = state["checks"][0]
        self.assertEqual(check["verdict"], "supported")
        self.assertFalse(is_displayable(check))
        self.assertEqual(after_verification(state), "analysis")
        state["revision_attempted"] = True
        self.assertEqual(after_verification(state), "explanation")
        self.assertEqual(
            explanation_agent(state)["answer"],
            "I couldn't find a supported answer in the retrieved passages.",
        )

    @patch("agent.verification_agent.chat")
    def test_supported_relevant_claim_stays_visible(self, mock_chat):
        mock_chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({
                "verdict": "supported", "answers_question": True, "reason": "Direct answer."
            }))
        )
        state = make_state(CLAIM)
        state.update(verification_agent(state))
        self.assertTrue(is_displayable(state["checks"][0]))
        self.assertEqual(after_verification(state), "explanation")
        self.assertIn(CLAIM, explanation_agent(state)["answer"])

    @patch("agent.verification_agent.chat")
    def test_missing_relevance_judgment_does_not_pass(self, mock_chat):
        mock_chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"verdict": "supported", "reason": "OK"}))
        )
        check = verification_agent(make_state(CLAIM))["checks"][0]
        self.assertFalse(is_displayable(check))
