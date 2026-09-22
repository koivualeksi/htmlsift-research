"""Gate: domain_of, the offline eTLD+1 grouping key the split folds on. Same
registrable domain across subdomains, scheme and case collapses to one key -- the
property that lets "no domain straddles the folds" hold. The fold assignment
itself (zero straddle, 545 domains / 732 pages, seed 20260815) reads the corpus
and is the full tier; this locks the grouping unit, offline.
"""
import pytest

from vendors.wmb.adapter import split


@pytest.mark.parametrize("url, expected", [
    ("https://www.example.com/path?q=1", "example.com"),
    ("http://blog.example.com", "example.com"),
    ("HTTPS://WWW.Example.COM", "example.com"),        # lowercased
    ("example.com/foo", "example.com"),                # no scheme
    ("https://example.co.uk/x", "example.co.uk"),      # multi-level public suffix
    ("https://shop.example.co.uk", "example.co.uk"),
    ("https://localhost/x", "localhost"),              # no registrable suffix
    ("", "unknown"),
])
def test_domain_of(url, expected):
    assert split.domain_of(url) == expected


def test_same_registrable_domain_collapses():
    urls = ["https://www.example.com/a", "http://blog.example.com",
            "https://a.b.example.com/x", "HTTPS://Example.COM"]
    assert len({split.domain_of(u) for u in urls}) == 1


def test_distinct_registrable_domains_separate():
    assert split.domain_of("https://example.com") != split.domain_of("https://example.co.uk")
