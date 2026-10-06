from onboardiq.generate.citations import extract_markers, format_citations, resolve_citations
from onboardiq.render import answer_markdown, escape
from onboardiq.types import Answer, Chunk, Citation, Hit


def _hits(n):
    return [Hit(Chunk(f"c{i}", "d", f"doc{i}.md", f"text {i}", heading=f"H{i}"), 1.0) for i in range(1, n + 1)]


def test_extract_markers_handles_groups_and_order():
    assert extract_markers("a [2] b [1][2] c [3, 1]") == [2, 1, 3]
    assert extract_markers("no citations") == []


def test_out_of_range_markers_are_removed():
    text, citations = resolve_citations("Fact one [1]. Fact two [4]. Both [1, 5].", _hits(2))
    assert text == "Fact one [1]. Fact two. Both [1]."
    assert [c.marker for c in citations] == [1]


def test_uncited_answer_still_lists_its_sources():
    text, citations = resolve_citations("An answer with no markers.", _hits(3))
    assert text == "An answer with no markers."
    assert [c.chunk_id for c in citations] == ["c1", "c2", "c3"]


def test_format_citations():
    rendered = format_citations([Citation(1, "c1", "doc1.md", "Guide > Setup", "...")])
    assert rendered == "Sources:\n[1] doc1.md > Guide > Setup (c1)"
    assert format_citations([]) == ""


def test_rendering_escapes_html_from_users_and_models():
    # problem 6: the prototype rendered questions and answers with unsafe_allow_html
    payload = '<script>alert("x")</script><img src=x onerror=1>'
    assert "<" not in escape(payload) and ">" not in escape(payload)
    answer = Answer(payload, payload, [Citation(1, "c1", "<b>doc</b>.md", "", "")], [], "data_analyst", "mid")
    rendered = answer_markdown(answer)
    assert "<script>" not in rendered and "<b>" not in rendered
    assert "&lt;script&gt;" in rendered
