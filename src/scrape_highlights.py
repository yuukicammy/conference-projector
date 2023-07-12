""" 
html_parser.py

Parses HTML and saves the list of papers to a database.
"""

import modal
from pathlib import Path
from typing import List, Any, Dict

from .config import ProjectConfig, Config

stub = modal.Stub(ProjectConfig._stub_highlights)
SHARED_ROOT = "/root/.cache"


def overwrite_image(
    config: Config, img_url: str | None, paper: Dict[str, str]
) -> Dict[str, str]:
    import requests

    if img_url in config.highlight.img_ignore_paths:
        img_url = None
    else:
        img_url = config.highlight.img_base_url + img_url

    if img_url is not None:
        response = requests.get(img_url)
        if response.status_code == 200:
            file_path = (
                Path(config.project.dataname)
                / "top_images"
                / (paper["id"].zfill(config.files.image_name_width) + ".png")
            )
            paper["image_path"] = str(file_path)
            (Path(SHARED_ROOT) / file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(Path(SHARED_ROOT) / file_path, "wb") as f:
                f.write(response.content)
    return paper


@stub.function(
    image=modal.Image.debian_slim().pip_install("beautifulsoup4", "requests"),
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=3600,
)
def scrape_award(config: Config) -> List[Dict[str, Any]]:
    import json
    from pathlib import Path
    import re
    from bs4 import BeautifulSoup
    import requests

    print(config.highlight.award_details_url)
    html = modal.Function.lookup(ProjectConfig._stub_scraper, "get_html").call(
        url=config.highlight.award_details_url
    )
    soup = BeautifulSoup(html, "html.parser")

    def scrape(award_title: str):
        highlight_divs = soup.find_all("div", text=award_title)
        for div in highlight_divs:
            title = div.find_next("a", class_="small-title").text
            img_url = div.find_next("img")["src"]
            print("Title:", title)
            print("Image URL:", img_url)
            items = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
                db_config=config.db,
                query=f'SELECT * FROM c WHERE c.title = "{title}"',
                force=True,
            )
            if items is not None and len(items) == 1:
                paper = items[0]
                paper["award"] = award_title
                paper = overwrite_image(config=config, img_url=img_url, paper=paper)
                modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
                    db_config=config.db, item=paper
                )

    scrape("Highlight")
    scrape("Award Candidate")


@stub.local_entrypoint()
def main(config_file: str = "configs/debug.toml"):
    """
    Main entry point of the script.

    Args:
        config_file (str): Path to the configuration file. Default is 'configs/defaults.toml'.
    """
    import json
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)
    scrape_award.call(config)
