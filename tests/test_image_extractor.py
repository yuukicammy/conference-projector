import unittest

from .test_common import PaperVizTestCase
from src.image_extractor import extract_representative_images


class TestExtractRepresentativeImages(PaperVizTestCase):
    def test_extract_representative_images(self):
        import copy
        import modal

        papers = copy.deepcopy(self.except_papers)
        papers[0]["arxiv_id"] = ""
        for p in papers:
            p["image_path"] = ""
        modal.Function.lookup(self.config.project._stab_db, "upsert_item").call(
            self.config.db, papers
        )

        extract_representative_images(self.config)
        actual_papers = modal.Function.lookup(
            self.config.project._stab_db, "get_all_papers"
        ).call(self.config.db, self.config.project.max_papers, force=True)
        assert len(papers) == len(self.except_papers)

        for actual, expected in zip(actual_papers, self.except_papers):
            actual["arxiv_id"] == expected["arxiv_id"]
            actual["image_path"] == expected["image_path"]
