import unittest

from evaluate_rag import build_bm25_retriever, tokenize


class BM25RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"text": "alpha berry recipe", "page": 1, "source": "example.pdf"},
            {"text": "raspberry orchard", "page": 2, "source": "example.pdf"},
            {"text": "cloud engine", "page": 3, "source": "example.pdf"},
        ]
        self.retrieve = build_bm25_retriever(self.chunks)

    def test_normalizes_case_and_hyphens(self):
        self.assertEqual(tokenize("ALPHA-berry"), ["alpha", "berry"])
        results = self.retrieve("ALPHA-berry")
        self.assertEqual([result["index"] for result in results], [0])
        self.assertEqual(results[0]["source"], "example.pdf")
        self.assertEqual(results[0]["page"], 1)

    def test_no_matching_tokens_returns_no_candidates(self):
        self.assertEqual(self.retrieve("ocean"), [])

    def test_candidate_limit(self):
        self.assertEqual(len(self.retrieve("orchard alpha", top_k=1)), 1)


if __name__ == "__main__":
    unittest.main()
