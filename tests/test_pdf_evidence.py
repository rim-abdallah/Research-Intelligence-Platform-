import unittest

import pymupdf

from pdf_evidence import render_evidence_page


class EvidenceHighlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Users join and leave during delivery.")
            page.insert_text((72, 99), "The paper ad-\nddressed a static scenario.")
            cls.pdf_bytes = pdf.tobytes()

    def test_short_quote_without_final_period(self):
        _, found = render_evidence_page(
            self.pdf_bytes, 1, "Users join and leave during delivery"
        )
        self.assertTrue(found)

    def test_quote_with_line_break_hyphen(self):
        _, found = render_evidence_page(
            self.pdf_bytes, 1, "The paper ad- ddressed a static scenario."
        )
        self.assertTrue(found)

    def test_absent_quote_is_not_highlighted(self):
        _, found = render_evidence_page(
            self.pdf_bytes, 1, "The users never leave."
        )
        self.assertFalse(found)
