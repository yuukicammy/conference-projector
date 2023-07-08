import unittest
from pathlib import Path

import modal

from src.config import ProjectConfig

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
)

SHARED_ROOT = "/root/.cache"

stub = modal.Stub(
    ProjectConfig._stab_test,
    image=modal_image,
    mounts=[
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "configs", remote_path="/root/configs"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "assets", remote_path="/root/src/assets"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "src", remote_path="/root/src"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "tests", remote_path="/root/tests"
        ),
    ],
    secrets=[
        modal.Secret.from_name("cosmos-secret"),
        modal.Secret.from_name("my-openai-secret"),
    ],
)


@stub.function(
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
    cpu=8,
)
def run_test():
    import subprocess

    response = subprocess.run(["python", "-m", "unittest"])
    print(response)


@stub.local_entrypoint()
def main():
    run_test.call()


if __name__ == "__main__":
    unittest.main()
