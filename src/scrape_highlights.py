""" 
html_parser.py

Parses HTML and saves the list of papers to a database.
"""

import modal
from typing import List, Any, Dict

from .config import ProjectConfig, Config

stub = modal.Stub(ProjectConfig._stub_highlights)
SHARED_ROOT = "/root/.cache"


@stub.function(
    image=modal.Image.debian_slim(),
    shared_volumes={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
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
        pattern_match(
            config.html_parser.prefix_item,
            config.html_parser.suffix_item,
            get_html(config.html_parser.base_url + config.html_parser.path_papers),
        )
    ):
        if config.project.max_papers <= len(papers):
            break
        print(path)
        paper_url = config.html_parser.base_url + path
        paper_html = get_html(url=paper_url)

        # find title
        title = pattern_match(
            prefix=config.html_parser.prefix_title,
            suffix=config.html_parser.suffix_title,
            string=paper_html,
            flag=re.DOTALL,
        )
        assert 0 < len(title)
        title = title[0].strip()  # remove the first and the last spaces
        assert 0 < len(title)

        # find abstract
        abstract = pattern_match(
            prefix=config.html_parser.prefix_abst,
            suffix=config.html_parser.suffix_abst,
            string=paper_html,
            flag=re.DOTALL,
        )
        assert 0 < len(abstract)
        abstract = abstract[0].strip()
        assert 0 < len(abstract)

        # find pdf url
        pdfurl = pattern_match(
            prefix=config.html_parser.prefix_pdf,
            suffix=config.html_parser.suffix_pdf,
            string=paper_html,
            flag=re.DOTALL,
        )
        assert 0 < len(pdfurl)
        pdfurl = pdfurl[0].strip()
        assert 0 < len(pdfurl)

        # find arxiv id
        arxiv_id = pattern_match(
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
        modal.Function.lookup(config.project._stub_db, "upsert_item").call(
            config.db, paper
        )

    if config.files.save_json:
        json_path = Path(SHARED_ROOT) / config.files.json_file
        with open(str(json_path), "w", encoding="utf-8") as fw:
            json.dump(papers, fw, indent=config.files.json_indent, ensure_ascii=False)
            print(f"Saved a json file: {json_path}")

    return papers


@stub.function()
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


@stub.function()
def make_pattern(prefix: str, suffix: str, reg: str = ".*") -> str:
    """
    Create a regular expressionpattern using the specified prefix and postfix.

    Args:
        prefix (str): Prefix of the pattern.
        postfix (str): Postfix of the pattern.
        reg (str): Regular expression pattern. Default is '.*'.

    Returns:
        str: Created regular expression pattern.
    """
    return prefix + reg + suffix


@stub.function()
def pattern_match(
    prefix: str, suffix: str, string, reg: str = ".*", flag: int = 0
) -> List[str]:
    """
    Perform pattern matching on the given string using the specified prefix and suffix.

    Args:
        prefix (str): Prefix of the pattern.
        suffix (str): Postfix of the pattern.
        string: String to perform pattern matching on.
        reg (str): Regular expression pattern. Default is '.*'.
        flag (int): Optional flag for pattern matching. Default is 0.

    Returns:
        List[str]: List of matched strings.
    """
    import re

    prefix = prefix.replace("\\n", "\n")
    suffix = suffix.replace("\\n", "\n")
    results = []
    for item in re.findall(
        pattern=make_pattern(prefix, suffix, reg), string=string, flags=flag
    ):
        item = item[len(prefix) : len(item) - len(suffix)]
        assert 0 < len(item)
        results.append(item)
    return results


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
