""" 
scraper.py

Scrape paper titles and related information.
"""

import modal
from typing import List, Dict, Any, Generator
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

    def search_arxiv(self, title: str) -> str:
        import arxiv

        arxiv_id = ""
        try:
            for res in arxiv.Search(
                query=f'{title.replace("-", " ")}'
            ).results():  # Do not search with "ti" because title search with "ti" will not find some titles.
                if (
                    res.title.replace("\n", "").replace(" ", "").lower()
                    == title.replace("\n", "").replace(" ", "").lower()
                ):
                    arxiv_id = res.entry_id.split("/")[-1]
                    arxiv_id = arxiv_id.split("v")[0]
                    break
        except arxiv.arxiv.UnexpectedEmptyPageError as e:
            print(e)
        finally:
            return arxiv_id

    def search_open_review(self, title: str) -> Dict[str, Any] | None:
        import requests
        from urllib.parse import urljoin

        base_url = "https://api2.openreview.net/notes/search"
        search_params = {
            "query": title.replace("/", "\/").replace("!", "").replace("-", " "),
            "limit": 10,  # Number of search results to retrieve (adjust as needed)
        }
        response = requests.get(url=base_url, params=search_params)
        if response.status_code == 200:
            search_results = response.json()
            for note in search_results["notes"]:
                # print(note["content"]["title"]["value"])
                if (
                    note["content"]["title"]["value"]
                    .replace("\n", "")
                    .replace(" ", "")
                    .lower()
                    == title.replace("\n", "").replace(" ", "").lower()
                ):
                    abstract = note["content"]["abstract"]["value"].strip()
                    pdf_path = note["content"]["pdf"]["value"].strip()
                    pdf_url = urljoin("https://openreview.net/", pdf_path)
                    return {"abstract": abstract, "pdf_url": pdf_url}
            return None
        else:
            print(
                f"Failed to retrieve search results. Status code: {response.status_code}, {response.json()}"
            )
            return None


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


class ICMLScraper(Scraper):
    def scrape(self) -> Generator[Dict[str, str], None, None]:
        """
        Scrape paper titles and related information from the ICML website.

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
        li_elements = soup.find_all("li", recursive=True)
        for li_element in li_elements:
            a_element = li_element.find("a")
            if (
                a_element
                and "href" in a_element.attrs
                and a_element["href"].startswith("/virtual/2023/poster/")
            ):
                title_path = a_element["href"]
                title = a_element.text.strip()
            else:
                continue

            # Scrape image url from the paper page.
            paper_html = self.get_html(
                url=urljoin(self.config.scraper.base_url, title_path)
            )
            paper_soup = BeautifulSoup(paper_html, "html.parser")

            # Find the <script> tag with type "application/ld+json"
            script_tag = paper_soup.find("script", type="application/ld+json")

            if script_tag:
                # Extract the content of the <script> tag as a JSON object
                script_content = json.loads(script_tag.string)

                # Extract the "contentUrl" from the JSON object
                content_url = script_content.get("contentUrl", None)

                image_url = ""
                if content_url:
                    if content_url not in self.config.scraper.img_ignore_paths:
                        image_url = urljoin(self.config.scraper.base_url, content_url)

            paper = {
                "url": urljoin(self.config.scraper.base_url, title_path),
                "pdf_url": urljoin(self.config.scraper.base_url, title_path),
                "arxiv_id": "",
                "title": title,
                "abstract": "",
                "award": "",
                "type": "",
                "image_url": image_url,
            }
            yield paper

    def scrape_orals(self) -> Generator[Dict[str, str], None, None]:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        oral_url = "https://icml.cc/virtual/2023/events/oral"
        paper_html = self.get_html(url=oral_url)
        soup = BeautifulSoup(paper_html, "html.parser")

        # Find all the <div> elements with class "displaycards touchup-date"
        paper_divs = soup.find_all(
            "div", class_="displaycards touchup-date", recursive=True
        )

        for paper_div in paper_divs:
            # Extract the title and URL
            title_elem = paper_div.find("a", class_="small-title")
            if title_elem:
                title = title_elem.text.strip()
            else:
                continue

            # Extract the abstract
            abstract_elem = paper_div.find("div", class_="abstract-display")
            if abstract_elem:
                abstract = abstract_elem.text.strip()
            else:
                abstract = "Abstract not available."

            img_elem = paper_div.find("img", class_="social-img-thumb")
            if img_elem and img_elem["src"] not in self.config.scraper.img_ignore_paths:
                img_url = urljoin(self.config.scraper.base_url, img_elem["src"])
            else:
                img_url = ""
            yield {
                "title": title,
                "abstract": abstract,
                "type": "Oral",
                "image_url": img_url,
            }

    def scrape_arxiv(self, title: str) -> Dict[str, str] | None:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        arxiv_id = self.search_arxiv(title=title)
        if len(arxiv_id) == 0:
            return None

        html = self.get_html(url=urljoin("https://arxiv.org/abs/", arxiv_id))
        soup = BeautifulSoup(html, "html.parser")
        abstract_meta = soup.find("meta", attrs={"name": "citation_abstract"})

        if abstract_meta:
            # Extract the content of the "citation_abstract" meta tag
            abstract = abstract_meta["content"].strip()
        else:
            abstract = ""
        pdf_url = urljoin("https://arxiv.org/pdf/", f"{arxiv_id}.pdf")

        return {"arxiv_id": arxiv_id, "abstract": abstract, "pdf_url": pdf_url}


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
    scraper = ICMLScraper(config=config)
    papers = []

    # Scrape all paper information one by one
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
    image=modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "beautifulsoup4",
        "requests",
        "arxiv@git+https://github.com/lukasschwab/arxiv.py.git",
    ),
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=36000,
    retries=0,
    cpu=8,
)
def pipline_icml_2023(
    config: Config,
    scrape_title: bool = False,
    scrape_orals: bool = False,
    scrape_arxiv: bool = False,
    scrape_open_review: bool = True,
):
    import concurrent

    papers = []
    scraper = ICMLScraper(config=config)
    idx = 0
    if scrape_title:
        # Scrape all paper information one by one
        for paper in scraper.scrape():
            if config.project.max_papers <= idx:
                break

            paper["id"] = str(idx)
            idx += 1
            print("Extracted: ", paper)
            papers.append(paper)

            # Save to DB
            res = modal.Function.lookup(config.project._stub_db, "upsert_item").call(
                db_config=config.db, item=paper
            )
            assert 6 < len(res.keys())
    if scrape_orals:
        # Update papers from oral pages.
        print("Scraping oral papers...")
        num_orals = 0
        for oral in scraper.scrape_orals():
            print(f"oral {num_orals}: ", oral)

            # Load the current information
            items = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
                db_config=config.db,
                query=f'SELECT * FROM c WHERE c.title = "{oral["title"]}"',
                force=True,
            )
            if items is None or len(items) == 0:
                continue

            paper = items[0]
            paper["abstract"] = oral["abstract"].strip()
            paper["type"] = "Oral"
            if 0 < len(oral["image_url"]):
                paper["image_url"] = oral["image_url"]

            # Update the paper information
            modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
                db_config=config.db, item=paper
            )
            num_orals += 1
        print(f"Total {num_orals} oral papers scraped.")

    num_papers = modal.Function.lookup(ProjectConfig._stub_db, "get_num_papers").call(
        db_config=config.db
    )

    if scrape_arxiv:
        print("Scraping arxiv information...")

        # Search arxiv and update paper information
        def update(idx: int) -> None:
            # Load the current information
            items = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
                db_config=config.db,
                query=f'SELECT * FROM c WHERE c.id = "{str(idx)}"',
                force=True,
            )
            if items is None or len(items) == 0:
                return
            paper = items[0]
            if (
                # 0 < len(#paper["arxiv_id"])
                # and
                paper["pdf_url"].startswith("https://arxiv.org")
                and 0 < len(paper["abstract"])
            ):
                # Arxiv info is already scraped.
                return
            arxiv_info = scraper.scrape_arxiv(title=paper["title"])
            print(f"arxiv info of  {idx}: ", arxiv_info)
            if arxiv_info is None:
                return
            for (
                key,
                value,
            ) in (
                arxiv_info.items()
            ):  # {"arxiv_id": arxiv_id, "abstract": abstract, "pdf_url": pdf_url}
                paper[key] = value

            # Update the paper information
            modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
                db_config=config.db, item=paper
            )

        with concurrent.futures.ThreadPoolExecutor(
            config.project.num_workers
        ) as executor:
            futures = [
                executor.submit(update, i)
                for i in range(min(num_papers, config.project.max_papers))
            ]
            concurrent.futures.wait(futures)

    if scrape_open_review:
        print("Searching Open Review...")

        # Load the current information
        def update(idx: int) -> None:
            items = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
                db_config=config.db,
                query=f'SELECT * FROM c WHERE c.id = "{str(idx)}"',
                force=True,
            )
            if items is None or len(items) == 0:
                return
            paper = items[0]
            if 0 < len(paper["abstract"]) and not paper["pdf_url"].startswith(
                "https://icml.cc"
            ):
                # Abstract already exists.
                return
            res = scraper.search_open_review(title=paper["title"])
            print(f"open review info of {idx}: ", res)
            if res is not None:
                paper["abstract"] = res["abstract"]
                paper["pdf_url"] = res["pdf_url"]
                # Update the paper information
                modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
                    db_config=config.db, item=paper
                )

        with concurrent.futures.ThreadPoolExecutor(
            config.project.num_workers
        ) as executor:
            futures = [
                executor.submit(update, i)
                for i in range(min(num_papers, config.project.max_papers))
            ]
            concurrent.futures.wait(futures)

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
    cpu=8,
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
def main(config_file: str = "configs/icml2023.toml"):
    """
    Main function to extract paper and award information from the web.

    Args:
        config_file (str): Path to the configuration file (defaults to "configs/defaults.toml").
    """
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)

    pipline_icml_2023.call(
        config=config,
        scrape_title=False,
        scrape_orals=False,
        scrape_arxiv=False,
        scrape_open_review=True,
    )
    # extract_awards_from_web.call(config=config)
    # extract_awards_from_config.call(config=config)
