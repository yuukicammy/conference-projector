import unittest
import modal

from .test_common import CPTestCase, stub, run_unittest_remote
from src.scraper import pattern_match, get_html


class TestGetHTML(CPTestCase):
    def test_get_html(self):
        url = "https://example.com/"
        expected = """<!doctype html>
<html>
<head>
    <title>Example Domain</title>

    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
        
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
        border-radius: 0.5em;
        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);
    }
    a:link, a:visited {
        color: #38488f;
        text-decoration: none;
    }
    @media (max-width: 700px) {
        div {
            margin: 0 auto;
            width: auto;
        }
    }
    </style>    
</head>

<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>
"""
        actual = get_html(url=url)
        assert expected == actual


class TestPatternMatch(CPTestCase):
    def test_pattern_match(self):
        string = "<a>test1</a>\n<a>test2</a>"
        actual = pattern_match(prefix="<a>", suffix="</a>", string=string)
        expected = ["test1", "test2"]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
