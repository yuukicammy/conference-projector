from typing import List
import unittest
import json
from pathlib import Path
import subprocess

import modal
import dacite
import toml

from src.config import Config, ProjectConfig

SHARED_ROOT = "/root/.cache"


modal_image = modal.Image.debian_slim().pip_install(
    "flask",
    "dacite",
    "dash",
    "toml",
    "dash-bootstrap-components",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "seaborn",
    "numpy",
    "azure-cosmos",
    "pymupdf",
    "Pillow",
    "requests",
    "pdf2image",
    "arxiv",
    "openai",
    "beautifulsoup4",
)

SHARED_ROOT = "/root/.cache"

stub = modal.Stub(
    ProjectConfig._stub_test,
    image=modal_image,
    mounts=[
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "assets", remote_path="/root/src/assets"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "src", remote_path="/root/src"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "tests", remote_path="/root/tests"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "configs", remote_path="/root/configs"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "data/prompts",
            remote_path="/root/data/prompts",
        ),
    ],
    secrets=[
        modal.Secret.from_name("cosmos-secret"),
        modal.Secret.from_name("my-openai-secret"),
    ],
)

if stub.is_inside():
    import os
    import dacite
    import toml
    import azure
    import azure.cosmos.cosmos_client as cosmos_client
    import azure.cosmos.exceptions as exceptions
    from azure.cosmos.partition_key import PartitionKey


class CPTestCase(unittest.TestCase):
    html_doc = "<a>test1</a>\n  <b>test2</b>"
    config = dacite.from_dict(data_class=Config, data=toml.load("configs/test.toml"))
    with open(Path(__file__).parent / "test.json", "r", encoding="utf-8") as f:
        expected_papers = json.load(f)


@stub.function(cpu=12)
def run_unittest_remote(files: List[str] | None = None) -> subprocess.CompletedProcess:
    cmd = ["python", "-m", "unittest"]
    if files is not None:
        cmd += files
    response = subprocess.run(cmd)
    return response


@stub.local_entrypoint()
def main():
    run_unittest_remote.call()
