import unittest

from .test_common import CPTestCase
from src.image_extractor import extract_representative_images
from src.cosmos import upsert_item, get_all_papers


class TestExtractRepresentativeImages(CPTestCase):
    def test_extract_representative_images(self):
        import copy

        papers = copy.deepcopy(self.expected_papers)
        papers[0]["arxiv_id"] = ""
        for idx, p in enumerate(papers):
            p["image_path"] = ""
            p["id"] = str(idx)
            upsert_item(self.config.db, p)

        extract_representative_images(self.config)
        actual_papers = get_all_papers(
            self.config.db, self.config.project.max_papers, force=True
        )
        assert len(papers) == len(self.expected_papers)

        for actual, expected in zip(actual_papers, self.expected_papers):
            actual["arxiv_id"] == expected["arxiv_id"]
            actual["image_path"] == expected["image_path"]
