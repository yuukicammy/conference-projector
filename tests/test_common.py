import unittest
from typing import Dict, Any
import json
from pathlib import Path

import dacite
import toml

from src.config import Config


class PaperVizTestCase(unittest.TestCase):
    html_doc = "<a>test1</a>\n  <b>test2</b>"
    config = dacite.from_dict(data_class=Config, data=toml.load("configs/test.toml"))
    with open(Path(__file__).parent / "test.json", "r", encoding="utf-8") as f:
        except_papers = json.load(f)
