import unittest
import modal

from .test_common import CPTestCase
from src.scraper import Scraper, CVPRScraper
from src.config import Config


class TestScraper(CPTestCase):
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
        actual = Scraper.get_html(url=url)
        assert expected == actual

    def test_not_implemented(self):
        assert Scraper(config=self.config).scrape() == NotImplemented


class TestCVPRScraper(CPTestCase):
    def test_scrape(self):
        scraper = CVPRScraper(config=self.config)
        expected_keys = sorted(
            ["url", "pdf_url", "arxiv_id", "title", "abstract", "award"]
        )
        num = 0
        for i, p in enumerate(scraper.scrape()):
            num += 1
            print("test_scrape", p)
            actual_keys = sorted(p.keys())
            assert actual_keys == expected_keys
            for key in expected_keys:
                assert p[key] == self.expected_papers[i][key]
        assert num == self.config.project.max_papers

    def test_scrape_awards(self):
        scraper = CVPRScraper(config=self.config)

        item1 = None
        for item in scraper.scrape_awards(
            config=self.config,
            award_titles=["Award Candidate"],
            invalid_paths=self.config.scraper.img_ignore_paths,
        ):
            if item["title"] == "What Can Human Sketches Do for Object Detection?":
                item1 = item

        assert item1["title"] == "What Can Human Sketches Do for Object Detection?"
        assert (
            item1["image_url"]
            == "https://cvpr2023.thecvf.com/media/PosterPDFs/CVPR%202023/22043-thumb.png?t=1685637488.807105"
        )
        assert item1["award"] == "Award Candidate"

        item2 = None
        for item in scraper.scrape_awards(
            config=self.config,
            award_titles=["Highlight"],
            invalid_paths=self.config.scraper.img_ignore_paths,
        ):
            if (
                item["title"]
                == "Hierarchical Dense Correlation Distillation for Few-Shot Segmentation"
            ):
                item2 = item
            if (
                item["title"]
                == "TarViS: A Unified Approach for Target-Based Video Segmentation"
            ):
                item3 = item

        assert (
            item2["title"]
            == "Hierarchical Dense Correlation Distillation for Few-Shot Segmentation"
        )
        assert item2["image_url"] == None
        assert item2["award"] == "Highlight"

        assert (
            item3["title"]
            == "TarViS: A Unified Approach for Target-Based Video Segmentation"
        )
        assert (
            item3["image_url"]
            == "https://cvpr2023.thecvf.com/media/PosterPDFs/CVPR%202023/23012-thumb.png?t=1685645785.7855182"
        )
        assert item3["award"] == "Highlight"


if __name__ == "__main__":
    unittest.main()
