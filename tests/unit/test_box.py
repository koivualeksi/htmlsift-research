"""Unit: gold-free box sizes (box_sizes), the rule the fp_box loss reads.

The page is the shared fixture of htmlsift-data's annotator, which marks the same
boxes; if this test and that one disagree, the annotator no longer shows what the
loss punishes. Gold is the h1 and the three plain article paragraphs.
"""
from vendors.wmb.adapter import blocks as wmb_blocks

PAGE = """<html><body>
  <header class="site-header">
    <div class="logo">Example News</div>
    <nav><ul><li>Home</li><li>News</li><li>Sport</li><li>Culture</li><li>Contact</li></ul></nav>
  </header>
  <main>
    <article>
      <h1 cc-select="true">Why the Helsinki tram line was delayed</h1>
      <p class="byline">By Aino Virtanen</p>
      <p cc-select="true">The city confirmed on Friday that the new line opens in March.</p>
      <p cc-select="true">Track work on the bridge took four months longer than planned.</p>
      <div class="share"><a>Facebook</a> <a>X</a> <a>Email</a></div>
      <p cc-select="true">Ticket prices stay at 3,10 euros for a single ride.</p>
    </article>
    <div><h3>Recent videos</h3><p>Snow arrives in Lapland</p><p>Ice hockey: Tappara wins</p>
      <p>Market hall reopens</p><p>Weather for the weekend</p></div>
  </main>
  <aside class="sidebar"><h3>Most read</h3>
    <ol><li>Ferry strike ends</li><li>New metro stations</li><li>Sauna rules change</li>
      <li>Elk on the ring road</li></ol></aside>
  <footer><p>© 2026 Example News</p><p>Privacy · Terms · Cookies</p>
    <p>Contact the newsroom</p></footer>
</body></html>"""


def _box_by_text():
    blocks, labels, box, _plain = wmb_blocks.label_page(PAGE)
    return {b.lstrip("-#0123456789. "): (y, s) for b, y, s in zip(blocks, labels, box)}


def test_largest_gold_free_container():
    got = _box_by_text()
    assert got["Example News"] == (0, 6)          # header, not the 5-line nav inside it
    assert got["Contact"] == (0, 6)
    assert got["Recent videos"] == (0, 5)         # a plain div counts, not only sections
    assert got["Weather for the weekend"] == (0, 5)
    assert got["Most read"] == (0, 5)
    assert got["Elk on the ring road"] == (0, 5)
    assert got["Contact the newsroom"] == (0, 3)  # below a threshold of 5, above 3


def test_inside_gold_container_is_small():
    got = _box_by_text()
    assert got["By Aino Virtanen"] == (0, 1)      # only its own p; the article has gold
    assert got["Facebook X Email"] == (0, 1)      # one line from three <a>: inside div.share
    assert got["Why the Helsinki tram line was delayed"] == (1, 0)


def test_no_gold_page_is_one_box():
    _, labels, box, _plain = wmb_blocks.label_page(PAGE.replace('cc-select="true"', ""))
    assert not any(labels) and set(box) == {len(box)}
