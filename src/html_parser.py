""" 
html_parser.py

Parses HTML and saves the list of papers to a database.
"""

import modal
from typing import List, Any, Dict

from .config import ProjectConfig, Config

stub = modal.Stub(ProjectConfig._stub_html_parser)
SHARED_ROOT = "/root/.cache"


@stub.function(
    image=modal.Image.debian_slim(),
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=3600,
    retries=0,
    cpu=1,
)
def parse_html(config: Config) -> List[Dict[str, Any]]:
    """
    Parse HTML and extract relevant information.

    Args:
        config (Config): Configuration object.

    Returns:
        List[Dict[str, Any]]: List of dictionaries with extracted information.
    """
    import json
    from pathlib import Path
    import re

    papers = []

    for idx, path in enumerate(
        modal.Function.lookup(ProjectConfig._stub_scraper, "pattern_match").call(
            config.html_parser.prefix_item,
            config.html_parser.suffix_item,
            modal.Function.lookup(ProjectConfig._stub_scraper, "get_html").call(
                config.html_parser.base_url + config.html_parser.path_papers
            ),
        )
    ):
        if config.project.max_papers <= len(papers):
            break
        paper_url = config.html_parser.base_url + path
        print(f"request url: {paper_url}")
        paper_html = modal.Function.lookup(
            ProjectConfig._stub_scraper, "get_html"
        ).call(url=paper_url)

        # find title
        title = modal.Function.lookup(
            ProjectConfig._stub_scraper, "pattern_match"
        ).call(
            prefix=config.html_parser.prefix_title,
            suffix=config.html_parser.suffix_title,
            string=paper_html,
            flag=re.DOTALL,
        )
        assert 0 < len(title)
        title = title[0].strip()  # remove the first and the last spaces
        assert 0 < len(title)

        # find abstract
        abstract = modal.Function.lookup(
            ProjectConfig._stub_scraper, "pattern_match"
        ).call(
            prefix=config.html_parser.prefix_abst,
            suffix=config.html_parser.suffix_abst,
            string=paper_html,
            flag=re.DOTALL,
        )
        assert 0 < len(abstract)
        abstract = abstract[0].strip()
        assert 0 < len(abstract)

        # find pdf url
        pdfurl = modal.Function.lookup(
            ProjectConfig._stub_scraper, "pattern_match"
        ).call(
            prefix=config.html_parser.prefix_pdf,
            suffix=config.html_parser.suffix_pdf,
            string=paper_html,
            flag=re.DOTALL,
        )
        assert 0 < len(pdfurl)
        pdfurl = pdfurl[0].strip()
        assert 0 < len(pdfurl)

        # find arxiv id
        arxiv_id = modal.Function.lookup(
            ProjectConfig._stub_scraper, "pattern_match"
        ).call(
            prefix=config.html_parser.prefix_arxiv,
            suffix=config.html_parser.suffix_arxiv,
            string=paper_html,
        )
        arxiv_id = "" if len(arxiv_id) == 0 else arxiv_id[0].strip()

        paper = {
            "url": paper_url,
            "pdf_url": pdfurl,
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
        }
        papers.append(paper)
        paper["id"] = str(idx)
        res = modal.Function.lookup(config.project._stub_db, "upsert_item").call(
            db_config=config.db, item=paper
        )
        assert 6 < len(res.keys())

    if config.files.save_json:
        json_path = Path(SHARED_ROOT) / config.files.json_file
        with open(str(json_path), "w", encoding="utf-8") as fw:
            json.dump(papers, fw, indent=config.files.json_indent, ensure_ascii=False)
            print(f"Saved a json file: {json_path}")

    return papers


@stub.local_entrypoint()
def main(config_file: str = "configs/defaults.toml"):
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
    papers = parse_html.call(config)
    with open("data/tmp.json", "w", encoding="utf-8") as fw:
        json.dump(papers, fw, indent=4, ensure_ascii=False)
