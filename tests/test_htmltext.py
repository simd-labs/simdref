"""Policy pins for the shared HTML stripper (spec 33, X7+D10)."""

from __future__ import annotations

from simdref.htmltext import squash, squeeze, strip_html, tidy, unwrap_anchors

# The RISC-V description policy: <br> and </p> become newlines, the inline
# markups drop, block containers split lines, everything else drops to a space.
RV_RULES = (
    ("br", (), "\n"),
    ("close", ("p",), "\n\n"),
    ("tags", ("code", "span", "strong", "em"), ""),
    ("tags", ("pre", "div", "section", "article", "ul", "ol", "li"), "\n"),
)


def test_tag_policies_replace_or_drop_by_class():
    assert strip_html("<code>x</code>", RV_RULES, generic=" ", ignorecase=True) == "x"
    assert strip_html("a<br>b", RV_RULES, generic=" ", ignorecase=True) == "a\nb"
    assert strip_html("a</p>b", RV_RULES, generic=" ", ignorecase=True) == "a\n\nb"
    assert strip_html("a<b>b", RV_RULES, generic=" ", ignorecase=True) == "a b"
    assert strip_html("a<B>b", RV_RULES, generic=" ", ignorecase=True) == "a b"
    # A stray < is text, not a tag.
    assert strip_html("a <> b", RV_RULES, generic=" ") == "a <> b"


def test_whitespace_tails_keep_their_site_shapes():
    # tidy: a whitespace run that holds a newline after its first character
    # collapses to one newline, and the tail after that newline stays.
    assert tidy("a \n b") == "a\n b"
    assert tidy("a\n\n\nb") == "a\nb"
    # squeeze keeps two newlines through whitespace noise; only 3+ consecutive
    # newlines collapse.
    assert squeeze("a\n\n \n\nb") == "a\n\n \n\nb"
    assert squeeze("a\n\n\nb") == "a\n\nb"
    assert squash("a\nb") == "a b"


def test_anchor_text_stays_without_the_tag():
    assert unwrap_anchors('x<a href="u">t</a>y') == "xty"
