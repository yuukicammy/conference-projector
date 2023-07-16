""" 
scraper.py

Scrape paper titles and related information.
"""

import modal
from typing import List, Dict, Generator
from pathlib import Path
import json

from .config import ProjectConfig, Config
from .utils import overwrite_image

stub = modal.Stub(ProjectConfig._stub_scraper)
SHARED_ROOT = "/root/.cache"


class Scraper:
    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def get_html(url: str) -> str:
        """
        Get the HTML content from the specified URL.

        Args:
            url (str): URL of the website.

        Returns:
            str: HTML content as a string.
        """

        import urllib

        response = urllib.request.urlopen(url=url)
        return response.read().decode()  # utf-8

    def scrape(self) -> NotImplemented:
        """
        Abstract method to be implemented by subclasses.
        Perform scraping of paper titles and related information.

        Returns:
            NotImplemented: This method is intended to be overridden by subclasses.
        """
        return NotImplemented


class CVPRScraper(Scraper):
    def scrape(self) -> Generator[Dict[str, str], None, None]:
        """
        Scrape paper titles and related information from the CVPR website.

        Yields:
            Dict[str, str]: Dictionary containing the scraped information for each paper.
        """
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        html = self.get_html(
            url=urljoin(self.config.scraper.base_url, self.config.scraper.path_papers)
        )
        soup = BeautifulSoup(html, "html.parser")

        # All titles
        title_elements = soup.find_all("dt", class_="ptitle")
        for idx, title_element in enumerate(title_elements):
            if self.config.project.max_papers <= idx:
                break
            title = title_element.a.text.strip()  # Title
            title_path = title_element.a["href"]  # Title URL, remove the first "/"

            dd_tags = title_element.find_next_siblings("dd")
            dd_tag = dd_tags[1]
            pdf_path = dd_tag.find("a", string="pdf")["href"]  # PDF URL
            arxiv_a = dd_tag.find("a", string="arXiv")
            if arxiv_a is not None:
                arxiv_url = arxiv_a["href"]  # arXiv URL
                arxiv_id = arxiv_url.split("/")[-1]  # arXiv ID (ex: "2212.08641")
            else:
                arxiv_id = ""

            # Abstract
            paper_html = self.get_html(
                url=urljoin(self.config.scraper.base_url, title_path)
            )
            paper_soup = BeautifulSoup(paper_html, "html.parser")
            abstract = paper_soup.select_one("#abstract").text.strip()

            paper = {
                "url": urljoin(self.config.scraper.base_url, title_path),
                "pdf_url": urljoin(self.config.scraper.base_url, pdf_path),
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "award": "",
            }
            yield paper

    def scrape_awards(
        self,
        config: Config,
        award_titles: List[str],
        invalid_paths: List[str] | None = None,
    ) -> Generator[Dict[str, str], None, None]:
        """
        Scrape paper awards from the CVPR website.

        Args:
            config (Config): Configuration object.
            award_titles (List[str]): List of award titles to scrape.
            invalid_paths (List[str] | None): List of URLs to ignore for the image path

        Yields:
            Dict[str, str]: Dictionary containing the scraped information for each award.
        """
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        print(config.scraper.award_details_url)
        html = self.get_html(url=config.scraper.award_details_url)
        soup = BeautifulSoup(html, "html.parser")

        for award_title in award_titles:
            highlight_divs = soup.find_all("div", string=award_title)
            for div in highlight_divs:
                title = div.find_next("a", class_="small-title").text
                img_path = div.find_next("img")["src"]
                if invalid_paths is not None and img_path in invalid_paths:
                    img_url = None
                else:
                    img_url = urljoin(config.scraper.img_base_url, img_path)
                print("Title:", title)
                print("Image URL:", img_url)
                yield {"title": title, "image_url": img_url, "award": award_title}


@stub.function(
    image=modal.Image.debian_slim().pip_install("beautifulsoup4", "requests"),
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=3600,
    retries=0,
    cpu=1,
)
def extract_papers_from_web(config: Config):
    """
    Extract paper information from the web and save it to the database and JSON file.

    Args:
        config (Config): Configuration object.
    """
    scraper = CVPRScraper(config=config)
    papers = []

    # Scrape paper information one by one
    for idx, paper in enumerate(scraper.scrape()):
        if config.project.max_papers <= idx:
            break

        paper["id"] = str(idx)
        print("Extracted: ", paper)
        papers.append(paper)

        # Save to DB
        res = modal.Function.lookup(config.project._stub_db, "upsert_item").call(
            db_config=config.db, item=paper
        )
        assert 6 < len(res.keys())

    # Save all paper info as a json file
    if config.files.save_json:
        json_path = Path(SHARED_ROOT) / config.files.json_file
        with open(str(json_path), "w", encoding="utf-8") as fw:
            json.dump(papers, fw, indent=config.files.json_indent, ensure_ascii=False)
            print(f"Saved a json file: {json_path}")


@stub.function(
    image=modal.Image.debian_slim().pip_install("beautifulsoup4", "requests"),
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=3600,
    retries=0,
    cpu=1,
)
def extract_awards_from_web(config: Config):
    """
    Extract award information from the web and update the corresponding papers in the database.

    Args:
        config (Config): Configuration object.
    """
    scraper = CVPRScraper(config=config)

    # Scrape paper information one by one
    for award_items in scraper.scrape_awards(
        config=config,
        award_titles=["Highlight", "Award Candidate"],
        invalid_paths=config.scraper.img_ignore_paths,
    ):
        title = award_items["title"]
        im_url = award_items["image_url"]
        award = award_items["award"]
        items = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
            db_config=config.db,
            query=f'SELECT * FROM c WHERE c.title = "{title}"',
            force=True,
        )
        if items is None or len(items) == 0:
            continue

        paper = items[0]
        paper["award"] = award

        if "image_path" not in paper.keys() or len(paper["image_path"]) == 0:
            paper["image_path"] = str(
                Path(config.project.dataname)
                / "top_images"
                / (paper["id"].zfill(config.files.image_name_width) + ".png")
            )
        overwrite_image(
            img_url=im_url,
            image_path=str(Path(SHARED_ROOT) / paper["image_path"]),
            img_ignore_paths=config.scraper.img_ignore_paths,
        )
        if not (Path(SHARED_ROOT) / paper["image_path"]).is_file():
            paper["image_path"] = ""

        modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
            db_config=config.db, item=paper
        )


@stub.function(
    image=modal.Image.debian_slim(),
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=3600,
)
def extract_awards_from_config(config: Config):
    """
    Extract award information from the configuration and update the corresponding papers in the database.

    Args:
        config (Config): Configuration object.
    """
    for item in config.scraper.award:
        award = item["label"]
        titles = item["values"]
        for title in titles:
            print(title)
            items = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
                db_config=config.db,
                query=f'SELECT * FROM c WHERE c.title = "{title}"',
                force=True,
            )
            if items is None or len(items) == 0:
                continue
            paper = items[0]
            paper["award"] = award
            print(paper)
            modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
                db_config=config.db, item=paper
            )


@stub.local_entrypoint()
def main(config_file: str = "configs/defaults.toml"):
    """
    Main function to extract paper and award information from the web.

    Args:
        config_file (str): Path to the configuration file (defaults to "configs/defaults.toml").
    """
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)

    extract_papers_from_web.call(config=config)
    extract_awards_from_web.call(config=config)
    extract_awards_from_config.call(config=config)
