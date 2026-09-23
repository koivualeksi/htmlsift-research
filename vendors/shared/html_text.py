"""HTML -> plain text for scoring an extractor's html-kind output on a text-metric
board (WCXB word-F1, DAnIEL ROUGE-L). trafilatura and readability emit main-content
HTML; these boards score words, so the html is flattened to text here -- the WCXB/DAnIEL
analogue of WMB's html2text inside score_text. One flattener, applied identically to
every html-kind extractor (ours included), so no extractor gets a different text path.
Not vendored: this conversion is ours, disclosed in docs/LIMITS.md.
"""
from lxml import html as lxml_html

# block-level tags whose boundaries must survive flattening: text_content() concatenates
# with no separator, which would fuse the last word of one block to the first of the next
# ("foo"+"bar" -> "foobar") and both metrics tokenize on word breaks. A trailing newline
# on each block keeps the tokens apart.
_BLOCK = {"p", "div", "br", "li", "tr", "td", "th", "section", "article", "header",
          "footer", "aside", "nav", "blockquote", "pre", "figure", "figcaption",
          "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6"}


def html_to_text(html: str) -> str:
    if not html or not html.strip():
        return ""
    root = lxml_html.fromstring(html)
    for bad in list(root.iter("script", "style")):
        bad.drop_tree()
    for el in root.iter():
        if el.tag in _BLOCK:
            el.tail = (el.tail or "") + "\n"
    lines = (ln.strip() for ln in root.text_content().splitlines())
    return "\n".join(ln for ln in lines if ln)
