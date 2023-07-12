""" 
scraper.py

Tools to scrape a HTML page.
"""

import modal
from typing import List, Any, Dict

from .config import ProjectConfig, Config

stub = modal.Stub(ProjectConfig._stub_scraper)
SHARED_ROOT = "/root/.cache"


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
    import json
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)
