"""Scanner tests for the regex-free display helpers (F17)."""

import unittest
from types import SimpleNamespace

from simdref.display import (
    _md_escape,
    _split_paragraphs,
    instruction_query_text,
    natural_query_sort_key,
    strip_instruction_decorators,
    strip_leading_tags,
)


class LeadingTagScannerTests(unittest.TestCase):
    def test_strips_the_whole_leading_tag_run(self):
        self.assertEqual(strip_leading_tags(" {evex} {load} vaddps x"), "vaddps x")

    def test_the_run_eats_the_whitespace_after_its_last_tag(self):
        self.assertEqual(strip_leading_tags("{evex}\t rest"), "rest")
        self.assertEqual(strip_leading_tags("{a} {}x"), "{}x")

    def test_a_run_without_a_tag_changes_nothing(self):
        self.assertEqual(strip_leading_tags("  {}x"), "  {}x")
        self.assertEqual(strip_leading_tags("}{evex}x"), "}{evex}x")

    def test_decorators_also_drop_the_evex_marker(self):
        self.assertEqual(strip_instruction_decorators(" {evex}vaddps _EVEX ymm"), "vaddps  ymm")


class ParagraphScannerTests(unittest.TestCase):
    def test_blank_line_shapes_split(self):
        self.assertEqual(_split_paragraphs("a\nb\n \n c"), ["a\nb", " c"])
        self.assertEqual(_split_paragraphs("a\n\n\nb"), ["a", "b"])

    def test_one_newline_stays_in_the_paragraph(self):
        self.assertEqual(_split_paragraphs("a\n b"), ["a\n b"])


class EscapeScannerTests(unittest.TestCase):
    def test_only_markdown_chars_grow_a_backslash(self):
        raw = "a`b_c[d]e<f>g|h#i~j\\k"
        self.assertEqual(_md_escape(raw), "a\\`b\\_c\\[d\\]e\\<f\\>g\\|h\\#i\\~j\\\\k")
        self.assertEqual(_md_escape("plain 42"), "plain 42")


class QueryScannerTests(unittest.TestCase):
    def test_tokens_split_on_space_comma_and_parens(self):
        item = SimpleNamespace(mnemonic="X", form="X (a(b), c)")
        self.assertEqual(instruction_query_text(item), "X a b c")
        item = SimpleNamespace(mnemonic="X", form="X (a,, b\tc)")
        self.assertEqual(instruction_query_text(item), "X a b c")

    def test_sort_key_splits_letter_and_number_runs(self):
        self.assertEqual(natural_query_sort_key("avx2x"), ((0, "avx"), (1, 2), (0, "x")))
        self.assertLess(natural_query_sort_key("M8"), natural_query_sort_key("M16"))

    def test_only_decimal_digits_form_number_runs(self):
        self.assertEqual(natural_query_sort_key("m\u0661\u0662"), ((0, "m"), (1, 12)))
        self.assertEqual(natural_query_sort_key("x\u00b2"), ((0, "x"),))
        self.assertEqual(natural_query_sort_key("\u0661\u00b2"), ((1, 1),))


if __name__ == "__main__":
    unittest.main()
